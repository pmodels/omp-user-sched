#ifndef _CKTASKQUEUE_H
#define _CKTASKQUEUE_H
#define TaskQueueSize 4096 
//Uncomment for debug print statements
#define TaskQueueDebug(...) //CmiPrintf(__VA_ARGS__)
// This taskqueue implementation refers to the work-stealing queue of Cilk (THE protocol).
// New tasks are pushed into the tail of this queue and the tasks are popped at the tail of this queue by the same thread. Thieves(other threads trying to steal) steal a task at the head of this queue.
// So, synchronization is needed only when there is only one task in the queue because thieves and victim can try to obtain the same task.
#include <iostream>
#include <atomic>
#include <array>
#include <algorithm>

//Single Producer-Multiple Consumer Queue
template <typename T> class TaskQueue {
  private: 
    std::atomic<int> head; // This pointer indicates the first task in the queue
    std::atomic<int> tail; // The tail indicates the array element next to the last available task in the queue. So, if head == tail, the queue is empty
    std::atomic<int> residual_head; // This pointer indicates the first task in the queue
    std::atomic<int> residual_tail; // The tail indicates the array element next to the last available task in the queue. So, if head == tail, the queue is empty

    //void *data[TaskQueueSize];
    std::array<T, TaskQueueSize> data;
    //std::atomic<int> wrap_around; // This value is 1 when producer inserts data into data[0:head-1].
    std::atomic<int> index; // index in the list of TaskQueue blocks. Used to prevent stealer and consumer cross over so each of them reach to beginning and end of the list
    std::atomic<int> residual_start; // Indicate whether producer starts consuming its own tasks
    std::atomic<int> fixed_block_size; // Producer set this value after it finishes to create tasks and  before it starts consuming the tasks. 
    int num_residual_tasks;
    TaskQueue<T> *next;
    TaskQueue<T> *prev;
  public:
    TaskQueue() {
      init();
    };

    void init(int index=0) {
      this->index.store(index, std::memory_order_relaxed);
      //wrap_around.store(0, std::memory_order_relaxed);
      //producer_consume_start.store(0, std::memory_order_relaxed);
      fixed_block_size.store(1, std::memory_order_relaxed);
      num_residual_tasks = 0;
      residual_head.store(0, std::memory_order_relaxed);
      residual_tail.store(0, std::memory_order_relaxed);
      residual_start.store(0, std::memory_order_relaxed);
      head.store(0, std::memory_order_relaxed);
      next = prev = NULL;
      tail.store(0, std::memory_order_release);
    };
    inline std::array<T, TaskQueueSize>& getData() {return data;} 
    inline int getHead() {return head.load(std::memory_order_relaxed);}
    inline int getTail() {return tail.load(std::memory_order_relaxed);}
    inline int getResHead() {return residual_head.load(std::memory_order_relaxed);}
    inline int getResTail() {return residual_tail.load(std::memory_order_relaxed);}
    inline int getNumResTasks() { return num_residual_tasks;}
    inline bool copyData(std::vector<T> *start, int start_idx, int insert_idx, int num_elements, int expected_num_threads) {
      if (num_elements > TaskQueueSize) {
        return false;
      } else {
        std::copy_n(start->begin()+start_idx, num_elements, data.begin()+insert_idx);
        tail.fetch_add(num_elements, std::memory_order_relaxed);
        return true;
      }
    }
    inline void setFixedBlockSize(int expected_num_threads, int new_tasks=0) { 
      int num_elements = new_tasks + tail.load(std::memory_order_relaxed) - head.load(std::memory_order_relaxed);
      int block_size = num_elements / expected_num_threads;
      if (block_size ==0)
        block_size = 1;
      fixed_block_size.store(block_size, std::memory_order_relaxed);

      if (block_size >1) 
        num_residual_tasks = num_elements % expected_num_threads;
      else 
        num_residual_tasks = 0;

      if (num_residual_tasks > 0) {
        residual_head.store(head.load(std::memory_order_relaxed)+num_elements-num_residual_tasks, std::memory_order_relaxed);
        residual_tail.store(head.load(std::memory_order_relaxed)+num_elements, std::memory_order_relaxed);
      }
      tail.store(head.load(std::memory_order_relaxed)+num_elements-num_residual_tasks, std::memory_order_relaxed);
      residual_start.store(0, std::memory_order_relaxed);
/*
      if (block_size > 1) {
        fixed_block_size.store(block_size, std::memory_order_relaxed);
        num_residual_tasks = num_elements % expected_num_threads;
        if (num_residual_tasks > 0) {
          residual_head.store(head.load(std::memory_order_relaxed)+block_size*expected_num_threads, std::memory_order_relaxed);
          residual_tail.store(tail.load(std::memory_order_relaxed), std::memory_order_relaxed);
          tail.store(head.load(std::memory_order_relaxed)+block_size*expected_num_threads, std::memory_order_relaxed);
        }
      }
      else  {
        fixed_block_size.store(1, std::memory_order_relaxed);
        num_residual_tasks = 0;
      }*/
    }
    inline void printData(int id) {
      for (int i= head ; i< tail ;i++) {
        printf("id: %d, data: %d, %d\n",id, data.at(i).lb, data.at(i).ub);
      }
    }
    inline void setNext(TaskQueue<T> *next) { this->next = next;}
    inline void setPrev(TaskQueue<T> *prev) { this->prev = prev;}
    inline void setIndex(int index) { this->index.store(index, std::memory_order_relaxed); }
    inline int getIndex() { return index.load(std::memory_order_relaxed); }
    inline TaskQueue<T> * getNext() { return next; }
    inline TaskQueue<T> * getPrev() { return prev; }
    inline bool isEmpty() { return (head.load(std::memory_order_relaxed) >= tail.load(std::memory_order_relaxed) && residual_head.load(std::memory_order_relaxed) >=residual_tail.load(std::memory_order_relaxed)); }
    bool enqueue(T input_data) {
      int cur_tail = tail.load(std::memory_order_relaxed);
      if (cur_tail >= TaskQueueSize) { // We pushed elements at the end of this taskqueue array-> This queue is full or the tail pointer reached the end of this queue
        // allocate new vector in the data;
        //next = new TaskQueue(this);
        return true;
      } else {
        data.at(cur_tail) = input_data;
      //  CmiMemoryWriteFence();
        //__atomic_thread_fence(__ATOMIC_RELEASE);
        tail.store((cur_tail+1), std::memory_order_release);
        return false;
      }
    };
    
    int enqueue_bulk(std::array<T, TaskQueueSize> *input_data, int start_idx, int num_tasks, int expected_num_threads) {
      int cur_tail = tail.load(std::memory_order_relaxed);
      if (cur_tail >= TaskQueueSize) { // We pushed elements at the end of this taskqueue array-> This queue is full or the tail pointer reached the end of this queue
        // allocate new vector in the data;
        //next = new TaskQueue(this);
        return num_tasks;
      } else {
        int num_avail_slot = TaskQueueSize - cur_tail;
        
        int left_tasks = num_tasks - num_avail_slot;
        int insert_tasks = 0;
        if (left_tasks <=0) {
          insert_tasks = num_tasks;
          left_tasks = 0;
        } else {
          insert_tasks = num_tasks - left_tasks;
        }

        std::copy_n(input_data->begin()+start_idx, insert_tasks, data.begin()+cur_tail);
        
        setFixedBlockSize(expected_num_threads, insert_tasks);
        std::atomic_thread_fence(std::memory_order_release);
        //tail.store((cur_tail+insert_tasks), std::memory_order_release);
        return left_tasks;
      }
    };

    bool try_dequeue(T* ptr) {
      int h, t;
      t = tail.load(std::memory_order_relaxed) -1;
      h = head.load(std::memory_order_acquire);
      if (t > h) { // This means there are more than two tasks in the queue, so it is safe to pop a task from the queue.
        *ptr = data.at(t);
        tail.store(t, std::memory_order_release);
        return true; 
      } else if (t < h) { // cur_wrap_around == 0 -> same as THE
        tail.store(h, std::memory_order_release);
        return false;
      }
      // From now on, we should handle the situation where there is only one task so thieves and victim can try to obtain this task simultaneously.
      tail.store((h + 1) % TaskQueueSize,std::memory_order_relaxed);

      if (!head.compare_exchange_strong(h, (h+1)%TaskQueueSize, std::memory_order_release, std::memory_order_relaxed)) //__sync_bool_compare_and_swap(&(Q->head), h, h+1)) //, 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) // Check whether the last task has already stolen.
        return false;
      *ptr = data.at(t);
      return true;  //data[t % TaskQueueSize];
    };

    inline std::array<T, TaskQueueSize> *try_dequeue_bulk(int *start_index, int *num_tasks, int nproc, int requested_num=-1) {
      int h, t;
      int expected_num_tasks; // = fixed_block_size.load(std::memory_order_relaxed); //requested_num;
      int divisor = nproc;
      std::atomic<int> *cur_tail;
      std::atomic<int> *cur_head;
      int residual_start_flag = residual_start.load(std::memory_order_relaxed);
      if (!residual_start_flag) {
        expected_num_tasks = fixed_block_size.load(std::memory_order_relaxed); 
        cur_tail = &tail;
        cur_head = &head;
      } else {
        cur_tail = &residual_tail;
        cur_head = &residual_head;
        expected_num_tasks =1;
      }

      t = cur_tail->load(std::memory_order_relaxed) - expected_num_tasks;
      h = cur_head->load(std::memory_order_acquire);
      if (h < t ) { // (t - expected_num_tasks)) { // Achieve the expected chunk
        *start_index = t; 
        *num_tasks = expected_num_tasks;
        cur_tail->store(*start_index, std::memory_order_release);
        return &data;
      } else if (t < h) {
        if (!residual_start_flag && num_residual_tasks > 0) {
          residual_start.store(1, std::memory_order_relaxed);
/*          expected_num_tasks =1;
          t = residual_tail.load(std::memory_order_relaxed);
          h = residual_head.load(std::memory_order_acquire);*/
        }

        cur_tail->store(h, std::memory_order_release);
        return NULL;
      }

      cur_tail->store(h+expected_num_tasks, std::memory_order_relaxed);
      if (!cur_head->compare_exchange_strong(h, h+expected_num_tasks, std::memory_order_release, std::memory_order_relaxed)) {
        if (!residual_start_flag && num_residual_tasks > 0) {
          residual_start.store(1, std::memory_order_release);
/*          expected_num_tasks =1;
          t = residual_tail.load(std::memory_order_relaxed);
          h = residual_head.load(std::memory_order_acquire);*/
        }
        return NULL;
      }
      else {
        *start_index = h;
        *num_tasks = expected_num_tasks; 
        return &data;
      }  
    };

    inline bool try_steal(T* ptr) {
      int h, t, next_head;
      while (1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        h = head.load(std::memory_order_relaxed);
        t = tail.load(std::memory_order_relaxed);
        if (h >= t) // The queue is empty or the last element has been stolen by other thieves or popped by the victim.
          return false;
        next_head = ( h+1 );
        if (!head.compare_exchange_weak(h, next_head, std::memory_order_release, std::memory_order_relaxed)) //  !__sync_bool_compare_and_swap(&(Q->head), h, h+1))//, 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) // Check whether the task this thief is trying to steal is still in the queue and not stolen by the other thieves.
          continue;
        *ptr =  data.at(h);
        return true;
      }
    };

    inline std::array<T, TaskQueueSize> *try_steal_bulk(int *start_index, int *num_tasks, int nproc, int requested_num=-1) { // similar to guided -> get # of left chunks / nproc -> reduce conflicts on heavy loaded taskqueue
      int h, t, cur_wrap_around, next_head;
      int expected_num_tasks = fixed_block_size.load(std::memory_order_acquire);
      int divisor = nproc;
      std::atomic<int> *cur_tail;
      std::atomic<int> *cur_head;

      while (1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (!residual_start.load(std::memory_order_relaxed)) {
          cur_tail = &tail;
          cur_head = &head;
          h = head.load(std::memory_order_relaxed);
          t = tail.load(std::memory_order_relaxed);
        } else {
          cur_tail = &residual_tail;
          cur_head = &residual_head;
          h = residual_head.load(std::memory_order_relaxed);
          t = residual_tail.load(std::memory_order_relaxed);
          expected_num_tasks = 1;
        }

        if (h >= t || h+expected_num_tasks >=t) // The queue is empty or the last element has been stolen by other thieves or popped by the victim.
          return NULL;

        if (!cur_head->compare_exchange_strong(h, h+expected_num_tasks, std::memory_order_release, std::memory_order_relaxed)) 
          return NULL;
        *start_index = h;
        *num_tasks = expected_num_tasks;
        return &data;
      }
    };
};
#endif
