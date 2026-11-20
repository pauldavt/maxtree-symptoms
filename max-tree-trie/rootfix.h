#include "include/common.h"

template <typename index_t>
class DynamicStack
{
public:
  DynamicStack(index_t* stack, size_t max_n) :
    stack_(stack), n_(0), max_n_(max_n), is_reallocated_(false)
  {    
  }

  ~DynamicStack()
  {
    if (is_reallocated_)
    {
      free(stack_);
    }
  }

  INLINE void insert(index_t x)
  {
    if (n_ >= max_n_)
    {
      reallocate();
    }

    stack_[n_++] = x;
  }

  INLINE index_t top() const
  {
    return stack_[n_ - 1U];
  }

  INLINE index_t remove()
  {
    return stack_[--n_];
  }

  NO_INLINE void reallocate()
  {
    if (!is_reallocated_)
    {    
      index_t* new_stack = (index_t*)malloc(2U * max_n_ * sizeof(index_t));
      assert(new_stack != nullptr);

      for (size_t i = 0; i < max_n_; ++i)
      {
        new_stack[i] = stack_[i];
      }

      stack_ = new_stack;
      max_n_ = 2U * max_n_;
      is_reallocated_ = true;
      return;
    }

    stack_ = (index_t*)realloc(stack_, 2U * max_n_ * sizeof(index_t));
    assert(stack_ != nullptr);
    max_n_ = 2U * max_n_;
  }

  INLINE size_t length() const
  {
    return n_;
  }

private:
  index_t* stack_;
  size_t n_;
  size_t max_n_;
  bool is_reallocated_;
};

template <typename index_t, typename attribute_t, typename functor1_t, typename functor2_t>
void rootfix_seq(index_t* parents, size_t n, attribute_t *RESTRICT attributes, functor1_t const& w, functor2_t const& plus)
{
  index_t *RESTRICT roots = new index_t[n];

  for (size_t i = 0; i < n; ++i)
  {
    roots[i] = parents[i];
    attributes[i] = w(i);
  }

  index_t on_stack[4096];
  DynamicStack<index_t> stack(&on_stack[0], 4096);

  for (size_t i = 0; i < n; ++i)
  {
    index_t root = i;

    while (root != roots[root])
    {
      stack.insert(root);
      root = roots[root];
    }

    attribute_t curr_attr = attributes[root];

    while (stack.length())
    {
      index_t current = stack.remove();
      curr_attr = plus(curr_attr, attributes[current]);
      attributes[current] = curr_attr;
      roots[current] = current;
    }
  }

  delete[] roots;
}