#include "include/timer.h"
#include "include/maxtree_trie.h"
#include "tests/random_elements.h"
#include <cassert>
#include <iomanip>

// for up to 4.2 gigapixel images, 32-bit indices can be used
using index_t = uint32_t;
// throughput decreases when using 64-bit indices
// using index_t = uint64_t;

// data type of pixel values
using value_t = float;

struct attr
{
  index_t area_;
  float mean_;
  float M2_;
};

double spectra[32][32];

int main()
{
  constexpr size_t width = 320U;
  constexpr size_t height = 320U;
  constexpr size_t depth = 1U;
  constexpr size_t n = width * height * depth;

  attr* attrs = new attr[n];
  value_t *values = new value_t[n];
  index_t *parents = new index_t[n];
  // index_t *areas = new index_t[n];

  std::default_random_engine generator;  
  mxt::random_elements(values, n, generator);

  // ensure that the node with index 0 is the root
  values[0] = std::numeric_limits<value_t>::min();

  // initial number of pixels in each set
  for (size_t i = 0; i < n; ++i)
  {
    attrs[i] = {1, values[i], 0.0f};
  }

  // when merging two pixel sets
  auto merge = [=](index_t to, index_t from) ALWAYS_INLINE
  {
    index_t n = attrs[to].area_ + attrs[from].area_;

    double delta = (double)attrs[from].mean_ - (double)attrs[to].mean_;
    double mu = (double)attrs[to].mean_ + delta * attrs[from].area_ / n;
    double M2 = (double)attrs[from].M2_ + (double)attrs[to].M2_ + delta * delta * attrs[to].area_ * attrs[from].area_ / n;

    attrs[to].M2_ = M2;
    attrs[to].mean_ = mu;
    attrs[to].area_ = n;
  };

  {
    mxt::Timer t;
    mxt::maxtree_trie(values, parents, merge, width, height);
  }  

  for (size_t y = 0; y < 32; ++y)
    for (size_t x = 0; x < 32; ++x)
    {
      spectra[y][x] = 0.0;
    }

  for (size_t i = 0; i < n; ++i)
  {
    int bin_area = roundf(log2f32(attrs[i].area_));
    int bin_mean = roundf(32 * attrs[i].mean_);
    if (bin_mean == 32) bin_mean = 31;

    spectra[bin_area][bin_mean] += attrs[i].area_ * (values[i] - values[parents[i]]) * attrs[i].mean_;
    //spectra[bin_area][bin_mean] += attrs[i].area_ * (values[i] - values[parents[i]]) * sqrtf(attrs[i].M2_ / (attrs[i].area_ - 1));        
  }

  for (size_t y = 0; y < 32; ++y)
  {
    for (size_t x = 0; x < 32; ++x)
    {
      printf("%f ", spectra[y][x]);    
    }
    printf("\n");
  }

  assert(attrs[0].area_ == n);

  delete[] attrs;
  delete[] parents;
  delete[] values;
}
