#define PY_ARRAY_UNIQUE_SYMBOL AN_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdint>
#include <cstring>

#include "include/maxtree_trie.h"
#include "rootfix.h"
#include "thread_pool.h"

using index_t = uint32_t;

static ThreadPool pool;


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Owns one Python reference and releases it when it goes out of scope.
class PyRef
{
public:
  explicit PyRef(PyObject* p = nullptr) : p_(p) {}
  ~PyRef() { Py_XDECREF(p_); }

  PyRef(const PyRef&) = delete;
  PyRef& operator=(const PyRef&) = delete;

  explicit operator bool() const { return p_ != nullptr; }
  PyArrayObject* arr() const { return (PyArrayObject*)p_; }
  PyObject* release() { PyObject* p = p_; p_ = nullptr; return p; }

private:
  PyObject* p_;
};

// Returns a C-contiguous, aligned array of the given dtype (copies only if needed).
static PyObject* as_array(PyObject* obj, int dtype)
{
  return PyArray_FROM_OTF(obj, dtype, NPY_ARRAY_IN_ARRAY);
}

template <typename T>
static T* data_ptr(const PyRef& a)
{
  return (T*)PyArray_DATA(a.arr());
}


// ---------------------------------------------------------------------------
// Max-tree attributes for a single image
// ---------------------------------------------------------------------------

// Builds the max-tree of `ch` (height x width) and computes, for every node,
// the area and the mean of the `n_data_ch` data channels over the node's pixels.
static void compute_attributes(const float* ch, size_t height, size_t width,
                               const float* data, size_t n_data_ch,
                               index_t* areas, float* out, index_t* parents)
{
  size_t n = height * width;

  for (size_t i = 0; i < n; ++i)
  {
    areas[i] = 1;
  }

  memcpy(out, data, n_data_ch * n * sizeof(float));

  // Merge pixel set `from` into `to`: running mean of the attributes.
  auto merge = [=](index_t to, index_t from) ALWAYS_INLINE
  {
    float* attr_to = out + n_data_ch * to;
    const float* attr_from = out + n_data_ch * from;

    index_t n_to = areas[to];
    index_t n_from = areas[from];
    index_t n_total = n_to + n_from;

    for (size_t k = 0; k < n_data_ch; ++k)
    {
      attr_to[k] += (n_from * (attr_from[k] - attr_to[k])) / n_total;
    }

    areas[to] = n_total;
  };

  mxt::maxtree_trie(const_cast<float*>(ch), parents, merge, width, height);
}


// ---------------------------------------------------------------------------
// attributes(channels, data) -> (out, areas, parents)
//
//   channels : float32 (B, H, W)
//   data     : float32 (B, H, W, C)
//
//   out      : float32 (B, H, W, C)  mean data per max-tree node
//   areas    : uint32  (B, H, W)     area per max-tree node
//   parents  : uint32  (B, H, W)     parent index per pixel, local to each image
// ---------------------------------------------------------------------------
static PyObject* attributes(PyObject*, PyObject* args)
{
  PyObject* arg_channels;
  PyObject* arg_data;

  if (!PyArg_ParseTuple(args, "OO", &arg_channels, &arg_data))
  {
    return NULL;
  }

  PyRef ch_arr(as_array(arg_channels, NPY_FLOAT32));
  if (!ch_arr) return NULL;

  PyRef data_arr(as_array(arg_data, NPY_FLOAT32));
  if (!data_arr) return NULL;

  if (PyArray_NDIM(ch_arr.arr()) != 3)
  {
    PyErr_SetString(PyExc_ValueError, "channels must have shape (B, H, W)");
    return NULL;
  }
  if (PyArray_NDIM(data_arr.arr()) != 4)
  {
    PyErr_SetString(PyExc_ValueError, "data must have shape (B, H, W, C)");
    return NULL;
  }

  const npy_intp* dims = PyArray_DIMS(ch_arr.arr());
  const npy_intp* dims_data = PyArray_DIMS(data_arr.arr());

  for (int d = 0; d < 3; ++d)
  {
    if (dims_data[d] != dims[d])
    {
      PyErr_SetString(PyExc_ValueError,
                      "channels (B, H, W) and data (B, H, W, C) shapes do not match");
      return NULL;
    }
  }

  const size_t batch_sz = (size_t)dims[0];
  const size_t height = (size_t)dims[1];
  const size_t width = (size_t)dims[2];
  const size_t n_data_ch = (size_t)dims_data[3];
  const size_t n = height * width;

  if (n > (size_t)UINT32_MAX)
  {
    PyErr_SetString(PyExc_ValueError, "image too large for 32-bit indices");
    return NULL;
  }

  npy_intp dims_out[4] = {dims[0], dims[1], dims[2], dims_data[3]};
  npy_intp dims_img[3] = {dims[0], dims[1], dims[2]};

  PyRef out_arr(PyArray_SimpleNew(4, dims_out, NPY_FLOAT32));
  if (!out_arr) return NULL;

  PyRef areas_arr(PyArray_SimpleNew(3, dims_img, NPY_UINT32));
  if (!areas_arr) return NULL;

  PyRef parents_arr(PyArray_SimpleNew(3, dims_img, NPY_UINT32));
  if (!parents_arr) return NULL;

  const float* ch = data_ptr<float>(ch_arr);
  const float* data = data_ptr<float>(data_arr);
  float* out = data_ptr<float>(out_arr);
  index_t* areas = data_ptr<index_t>(areas_arr);
  index_t* parents = data_ptr<index_t>(parents_arr);

  if (batch_sz > 0 && n > 0)
  {
    // The workers make no Python API calls, so the GIL can be released.
    Py_BEGIN_ALLOW_THREADS

    pool.for_all_blocks(batch_sz, [=](size_t i, thread_nr_t) {
      compute_attributes(ch + i * n, height, width,
                         data + i * n * n_data_ch, n_data_ch,
                         areas + i * n,
                         out + i * n * n_data_ch,
                         parents + i * n);
    });

    Py_END_ALLOW_THREADS
  }

  // "N" steals the references.
  return Py_BuildValue("NNN", out_arr.release(), areas_arr.release(), parents_arr.release());
}


// ---------------------------------------------------------------------------
// max_ancestor_filter(parents, values) -> out
//
//   parents  : uint32  (B, H, W)  parent index per pixel, local to each image
//                                 (as returned by attributes)
//   values   : float32 (B, H, W)  value per max-tree node
//
//   out      : float32 (B, H, W)  for every node, the maximum of `values`
//                                 over the node and all its ancestors
// ---------------------------------------------------------------------------
static PyObject* max_ancestor_filter(PyObject*, PyObject* args)
{
  PyObject* arg_parents;
  PyObject* arg_values;

  if (!PyArg_ParseTuple(args, "OO", &arg_parents, &arg_values))
  {
    return NULL;
  }

  PyRef parents_arr(as_array(arg_parents, NPY_UINT32));
  if (!parents_arr) return NULL;

  PyRef values_arr(as_array(arg_values, NPY_FLOAT32));
  if (!values_arr) return NULL;

  if (PyArray_NDIM(parents_arr.arr()) != 3 || PyArray_NDIM(values_arr.arr()) != 3)
  {
    PyErr_SetString(PyExc_ValueError, "parents and values must have shape (B, H, W)");
    return NULL;
  }

  const npy_intp* dims = PyArray_DIMS(values_arr.arr());
  const npy_intp* dims_parents = PyArray_DIMS(parents_arr.arr());

  for (int d = 0; d < 3; ++d)
  {
    if (dims_parents[d] != dims[d])
    {
      PyErr_SetString(PyExc_ValueError, "parents and values shapes do not match");
      return NULL;
    }
  }

  const size_t batch_sz = (size_t)dims[0];
  const size_t n = (size_t)dims[1] * (size_t)dims[2];

  if (n > (size_t)UINT32_MAX)
  {
    PyErr_SetString(PyExc_ValueError, "image too large for 32-bit indices");
    return NULL;
  }

  npy_intp dims_out[3] = {dims[0], dims[1], dims[2]};
  PyRef out_arr(PyArray_SimpleNew(3, dims_out, NPY_FLOAT32));
  if (!out_arr) return NULL;

  index_t* parents = data_ptr<index_t>(parents_arr);
  const float* values = data_ptr<float>(values_arr);
  float* out = data_ptr<float>(out_arr);

  if (batch_sz > 0 && n > 0)
  {
    // The workers make no Python API calls, so the GIL can be released.
    Py_BEGIN_ALLOW_THREADS

    pool.for_all_blocks(batch_sz, [=](size_t i, thread_nr_t) {
      const float* values_i = values + i * n;

      auto const& max_op = [](float a, float b) ALWAYS_INL_L(float)
      {
        return a > b ? a : b;
      };

      auto const& weight = [=](index_t j) ALWAYS_INL_L(float)
      {
        return values_i[j];
      };

      rootfix_seq(parents + i * n, n, out + i * n, weight, max_op);
    });

    Py_END_ALLOW_THREADS
  }

  return out_arr.release();
}


// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

static PyMethodDef methods[] =
{
  {"max_ancestor_filter", max_ancestor_filter, METH_VARARGS, nullptr},
  {"attributes",          attributes,          METH_VARARGS, nullptr},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module =
{
  PyModuleDef_HEAD_INIT,
  "mta",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_mta(void)
{
  import_array();
  return PyModule_Create(&module);
}