#pragma once

#include <pthread.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <cstring>

#include "include/common.h"

constexpr size_t default_max_threads_limit = 256;
constexpr size_t default_n_items_per_block = 64U * 1024U;
constexpr size_t cacheline_len = 64U;

using dim_idx_t = uint_fast8_t;
using thread_nr_t = uint_fast16_t;

template <
  typename Index,
  typename Value = Index,
  size_t NDimensions = 1,
  size_t NNeighbors = 2 * NDimensions,
  std::enable_if_t<
    std::is_integral<Index>::value &&
    std::is_unsigned<Index>::value &&
    (std::is_integral<Value>::value ||
    std::is_floating_point<Value>::value), bool> = true>
struct primitives
{
  using index_t = Index;
  using value_t = Value;

  static constexpr size_t n_dimensions = NDimensions;
  static constexpr size_t n_neighbors = NNeighbors;
};

template <typename A, typename B>
constexpr A div_roundup(A const& a, B const& b)
{
  A result = a / b;

  if (a % b)
  {
    ++result;
  }

  return result;
}

template <typename Sub>
struct Range
{
  using sub_t = Sub;

  sub_t begin_;
  sub_t end_;
};

template <size_t NDimensions>
class Dimensions
{
public:
  static constexpr size_t n_dimensions = NDimensions;

  constexpr size_t length() const
  {
    size_t x = xs_[0];

    for (size_t d = 1; d < n_dimensions; ++d)
    {
      x *= xs_[d];
    }

    return x;
  }

  constexpr size_t& operator[](size_t d)
  {
    return xs_[d];
  }

  constexpr size_t const& operator[](size_t d) const
  {
    return xs_[d];
  }

  size_t xs_[n_dimensions];
};


template <typename Primitives>
class Coordinate
{
public:
  using prim = Primitives;
  using index_t = typename prim::index_t;
  using dim_t = Dimensions<prim::n_dimensions>;

  static constexpr size_t n_dimensions = prim::n_dimensions;

  static Coordinate from_index(index_t index, dim_t const& dims);

  constexpr index_t index(dim_t const& dims) const
  {
    index_t result = 0;
    index_t factor = 1;

    for (dim_idx_t d = 0; d != n_dimensions; ++d)
    {
      result += factor * xs_[d];      
      factor *= dims[d];
    }

    return result;    
  }
  
  constexpr index_t& operator[](size_t d)
  {
    return xs_[d];
  }

  constexpr index_t const& operator[](size_t d) const
  {
    return xs_[d];
  }  

  void init_zeros();
  void inc_index(dim_t const& dims);
   
  index_t xs_[n_dimensions];
};

template <typename prim>
Coordinate<prim> Coordinate<prim>::from_index(index_t index, dim_t const& dims)
{
  Coordinate result;

  for (dim_idx_t d = 0; d != n_dimensions; ++d)
  {
    result.xs_[d] = index % dims[d];
    
    index /= dims[d];
  }

  return result;
}

template <typename prim>
void Coordinate<prim>::init_zeros()
{
  for (dim_idx_t d = 0; d < n_dimensions; ++d)
  {
    xs_[d] = 0;
  }
}

template <typename prim>
void Coordinate<prim>::inc_index(dim_t const& dims)
{
  ++xs_[0];
  for (dim_idx_t d = 1; d < n_dimensions; ++d)
  {
    if (xs_[d - 1] == dims[d - 1])
    {
      xs_[d - 1] = 0;
    }
    else if (xs_[d - 1] != 0)
    {
      break;
    }

    ++xs_[d];
  }
}

class ThreadPool
{
public:
  using range_t = Range<size_t>;

  ThreadPool();
  ~ThreadPool();
  
  void parallel(
    void (*user_f)(void *, thread_nr_t thread_nr),
    void *user_data);

  template <typename functor_t>
  void for_all_blocks(size_t n_blocks, functor_t const &user_block_f);

  template <typename primitives_t, typename functor_t>
  void for_all_blocks(
    Dimensions<primitives_t::n_dimensions> const &dims,
    functor_t const &user_block_f);

  template <typename functor_t>
  void for_all(
    size_t n,
    functor_t const &user_f,
    size_t n_items_per_block = default_n_items_per_block);

  constexpr size_t max_threads() const { return max_threads_; }

  constexpr size_t n_active_threads() const { return n_active_threads_; }  

  constexpr void set_n_active_threads(size_t n)
  {
    if (n == 0 || n > max_threads_)
    {
      n_active_threads_ = max_threads_;
      return;
    }

    n_active_threads_ = n;
  }  

private:
  enum state {running, terminating};

  struct ThreadDataUnpadded
  {
    std::atomic<Range<size_t>> range_;
    ThreadPool* pool_;
    pthread_t id_;
    bool flag_;
    thread_nr_t thread_nr_;    
  };

  struct ThreadData
  {
    union
    {
      ThreadDataUnpadded data_;
      // avoid false sharing
      uint8_t data_plus_padding_[
        ((sizeof(ThreadDataUnpadded) + cacheline_len - 1U) / cacheline_len) *
        cacheline_len];
    };
  };

  void wake_threads();
  void wait_ready();
  void create_threads();
  void join_threads();
  
  template <typename functor_t>
  void iterate_range(
    size_t begin,
    size_t end,
    functor_t const &user_elem_f,
    thread_nr_t thread_nr);

  static bool find_work(ThreadData* td, thread_nr_t thread_nr, size_t n_threads, size_t* block_nr);
  static void* thread_pool_loop(void* data);

  size_t max_threads_;  
  size_t n_active_threads_;
  ThreadData* thread_data_;
  pthread_mutex_t mutex_ = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t cond_ = PTHREAD_COND_INITIALIZER;
  bool main_flag_{false};
  std::atomic<size_t> n_ready_{0U};
  state state_; // only changed by the main thread and if the mutex is locked
  void (*user_f_)(void*, size_t){nullptr};
  void* user_data_{nullptr};
  cpu_set_t cpu_set_;
  pthread_attr_t pthread_attr_;  
};

template <typename functor_t>
void ThreadPool::for_all_blocks(size_t n_blocks, functor_t const &user_block_f)
{
  using prim = primitives<size_t>;
  using dim_t = Dimensions<prim::n_dimensions>;
  using vec_t = Coordinate<prim>;

  for_all_blocks<prim>(dim_t({n_blocks}),
    [=](vec_t const& b, thread_nr_t thread_nr) ALWAYS_INLINE
    {
      user_block_f(b[0], thread_nr);
    });
}

template <typename prim, typename functor_t>
void ThreadPool::for_all_blocks(
  Dimensions<prim::n_dimensions> const &grid_dims,
  functor_t const &user_block_f)
{   
  using vec_t = Coordinate<prim>;
  using index_t = typename prim::index_t;

  struct SharedData
  {
    functor_t const &user_block_f_;
    Dimensions<prim::n_dimensions> const grid_dims_;
    ThreadData* thread_data_;
    size_t n_threads_;
  };

  size_t n_blocks = grid_dims.length();

  if (n_blocks == 0U)
  {
    return;
  }

  if (n_blocks == 1U)
  {
    vec_t v = vec_t::from_index(index_t(0), grid_dims);
    user_block_f(v, 0U);
    return;
  }

  if (n_active_threads_ == 1U)
  {
    vec_t v = vec_t::from_index(index_t(0), grid_dims);

    size_t begin = 0;
    size_t end = grid_dims.length();

    for (; begin < end; ++begin)
    {
      user_block_f(v, 0U);
      v.inc_index(grid_dims);
    }

    return;
  }

  size_t n_threads = std::min(n_active_threads_, n_blocks);

  SharedData data{user_block_f, grid_dims, thread_data_, n_threads};

  size_t per_thread = n_blocks / n_threads;
  size_t remainder = n_blocks % n_threads;
  size_t begin = 0;
  Range<size_t> range;
  
  for (size_t i = 0; i < n_threads; ++i)
  {
    range.begin_ = begin;
    begin += per_thread + size_t(i < remainder);
    range.end_ = begin;
    thread_data_[i].data_.range_.store(range, std::memory_order_relaxed);
  }

  user_f_ = [](void *data_p, thread_nr_t thread_nr)
  {      
    SharedData& data = *reinterpret_cast<SharedData *>(data_p);

    if (thread_nr >= data.n_threads_)
    {
      return;
    }

    size_t block_nr = 0U;

    while (true)
    {
      if (!find_work(data.thread_data_, thread_nr, data.n_threads_, &block_nr))
      {
        return;
      }

      vec_t v = vec_t::from_index(block_nr, data.grid_dims_);
      data.user_block_f_(v, thread_nr);
    }
  };

  parallel(user_f_, &data);
}

template <typename functor_t>
NO_INLINE void ThreadPool::iterate_range(
  size_t begin,
  size_t end,
  functor_t const &user_elem_f,
  thread_nr_t thread_nr)
{    
  for (; begin < end; ++begin)
  {
    user_elem_f(begin, thread_nr);
  }
}

template <typename functor_t>
void ThreadPool::for_all(
  size_t n,
  functor_t const& user_f,
  size_t n_items_per_block)
{
  size_t n_blocks = div_roundup(n, n_items_per_block);

  for_all_blocks(n_blocks, [=](size_t block_nr, thread_nr_t thread_nr) {
    size_t begin = block_nr * n_items_per_block;
    size_t end = begin + n_items_per_block;

    if (block_nr == n_blocks - 1U)
    {
      end = n;
    }

    iterate_range(begin, end, user_f, thread_nr);
  });
}

inline void ThreadPool::wake_threads()
{
  // mutex locked
  pthread_mutex_unlock(&mutex_);
  pthread_cond_broadcast(&cond_);
  // mutex unlocked
}

inline void ThreadPool::wait_ready()
{
  // mutex unlocked
  while (n_ready_.load(std::memory_order_consume) != max_threads_ - 1U)
  {
    // busy wait
  }

  pthread_mutex_lock(&mutex_);  
  // all other threads are waiting on the condition var
  n_ready_.store(0, std::memory_order_relaxed);
  // mutex locked
}

inline void* ThreadPool::thread_pool_loop(void* data)
{
  ThreadDataUnpadded &td = *reinterpret_cast<ThreadDataUnpadded*>(data);
  ThreadPool &tp = *td.pool_;

  while (true)
  {
    pthread_mutex_lock(&tp.mutex_);

    tp.n_ready_.fetch_add(1U, std::memory_order_release);

    do
    {      
      if (tp.state_ == ThreadPool::terminating)
      {
        pthread_mutex_unlock(&tp.mutex_);
        return nullptr;
      }

      pthread_cond_wait(&tp.cond_, &tp.mutex_);
    } while (tp.main_flag_ == td.flag_); // avoid spurious wakeup

    pthread_mutex_unlock(&tp.mutex_);

    td.flag_ = !td.flag_;

    if (td.thread_nr_ >= tp.n_active_threads_)
    {
      continue;
    }

    tp.user_f_(tp.user_data_, td.thread_nr_);
  }
}

inline void ThreadPool::create_threads()
{
  ThreadDataUnpadded &td = thread_data_[0].data_;
  td.thread_nr_ = 0;
  td.pool_ = this;
  td.id_ = pthread_self();
  td.flag_ = false;

  // outside a parallel operation, thread 0 can use all cores.
  // in a parallel operation, thread 0 has an affinity for core 0.

  for (size_t i = 1; i < max_threads_; ++i)
  {   
    ThreadDataUnpadded &td = thread_data_[i].data_;
    td.thread_nr_ = i;
    td.pool_ = this;
    td.flag_ = false;

    CPU_ZERO(&cpu_set_);
    CPU_SET(i, &cpu_set_);
    pthread_attr_setaffinity_np(&pthread_attr_, sizeof(cpu_set_t), &cpu_set_);
    
    pthread_create(
      &td.id_,
      &pthread_attr_,
      thread_pool_loop,
      &td);
  }  
}

inline void ThreadPool::join_threads()
{
  for (size_t i = 1; i < max_threads_; ++i)
  {   
    ThreadDataUnpadded &td = thread_data_[i].data_;
    pthread_join(td.id_, nullptr);
  }       
}

inline void ThreadPool::parallel(void (*user_f)(void *, size_t), void *user_data)
{
  // mutex locked 

  cpu_set_t saved;
  pthread_getaffinity_np(thread_data_[0].data_.id_, sizeof(cpu_set_t), &saved);
  CPU_ZERO(&cpu_set_);
  CPU_SET(0U, &cpu_set_);
  pthread_setaffinity_np(thread_data_[0].data_.id_, sizeof(cpu_set_t), &cpu_set_);
  memcpy(&cpu_set_, &saved, sizeof(cpu_set_t));

  main_flag_ = !main_flag_;
  user_data_ = user_data;
  user_f_ = user_f;
  
  wake_threads();    
  // mutex unlocked

  user_f_(user_data, 0U);

  wait_ready();
  // mutex locked

  pthread_setaffinity_np(thread_data_[0].data_.id_, sizeof(cpu_set_t), &cpu_set_);
}  

inline bool ThreadPool::find_work(
  ThreadData* tds,
  thread_nr_t thread_nr,
  size_t n_threads,
  size_t* block_nr)
{
  ThreadDataUnpadded& td = tds[thread_nr].data_;
  range_t current = td.range_.load(std::memory_order_relaxed);
  range_t desired;

  do 
  {
    if (current.begin_ >= current.end_)
    {
      thread_nr_t k = thread_nr + 1U;

      if (k == n_threads)
      {
        k = 0U;
      }

      size_t i = 0;
      for (; i < n_threads - 1U; ++i)
      {
        if (k == thread_nr)
        {
          continue;
        }

        ThreadDataUnpadded& other = tds[k].data_;

        range_t other_range = other.range_.load(std::memory_order_relaxed);
        range_t other_desired;
        size_t length;

        do
        {
          length = other_range.end_ - other_range.begin_;

          if (length == 0U)
          {
            break;
          }

          size_t half = (length + 1U) / 2U;
          other_desired = {other_range.begin_, other_range.end_ - half};
        } while (!other.range_.compare_exchange_weak(
            other_range,
            other_desired,
            std::memory_order_relaxed));

        if (length > 0U)
        {
          *block_nr = other_desired.end_;
          range_t new_range{other_desired.end_ + 1U, other_range.end_};
          td.range_.store(new_range, std::memory_order_relaxed);

          return true;          
        }

        ++k;
        if (k == n_threads)
        {
          k = 0U;
        }
      }

      if (i == n_threads - 1U)
      {
        return false;
      }
    }

    desired = {current.begin_ + 1U, current.end_};
  } while (!td.range_.compare_exchange_weak(current, desired, std::memory_order_relaxed));     

  *block_nr = current.begin_;
  return true;
}

inline ThreadPool::ThreadPool() :
  state_(running)
{
  max_threads_ = std::min(
    std::max(size_t(std::thread::hardware_concurrency()), size_t(1)),
  default_max_threads_limit);

  n_active_threads_ = max_threads_,

  thread_data_ = reinterpret_cast<ThreadData*>(
    aligned_alloc(cacheline_len, sizeof(ThreadData) * max_threads_));

  pthread_attr_init(&pthread_attr_); 

  create_threads();
  wait_ready();
  // mutex locked
}

inline ThreadPool::~ThreadPool()
{
  // mutex locked
  state_ = terminating;
  wake_threads();
  join_threads();

  free(thread_data_);
  pthread_attr_destroy(&pthread_attr_); 
}
