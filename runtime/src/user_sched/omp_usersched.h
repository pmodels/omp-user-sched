#ifndef __KMP_INTEROP_H__
#define __KMP_INTEROP_H__
typedef struct thread_bind_config {
  int isPinned; // The thread or task can be migrated from the initial placement - 0: non-migratable, 1:migratable
  int isGlobal; // The configuration is set as a global configuration - 1: The configuration is applied for all the upcoming OpenMP regions, 0: Temporarily used
  void *param; // User can pass parameters here - user can schedule threads with specific charateristics(e.g. MPI calls) or schedule threads using the data passed by this pointer (e.g. using historical info to decide where to put threads  
} thread_bind_config;

/* set schedulers */
#if __cplusplus
extern "C" {
#endif
/* Thread schedulers 
 * This function set a user defined function which distributes threads across cores or threads.
 * Users can change the distribution of threads temporarily or permanently.
 * */
void ompx_set_usersched_thread_bind(int start, int end, thread_bind_config config, int (*func)(int id, thread_bind_config)); // From thread[start] to thread[end], the user-defined scheduler is applied
void ompx_set_usersched_thread_bind_explicit(int *thread_id_array, thread_bind_config config, int (*func)(int id, thread_bind_config)); // User can pass a list of threads which the user defined scheduler will be applied

void ompx_set_usersched_for_loops(void (*chunks_divide_func)(int left_start, int left_end, int *assigned_start, int *assigned_end), int (*chunks_place_func)(int start_iter, int end_iter));

enum task_type {NO_DEPEND, DEPEND, TASKLOOP};

typedef struct usersched_task_info {
  enum task_type cur_task_type;
  int scheduled_tid;
  int label;
  int level;
  int loop_start;
  int loop_end;
  struct usersched_task_info *parent_info;
  struct usersched_task_info *depend_info;
} usersched_task_info;

void ompx_set_usersched_task_sched(const char *label, int (*task_sched)(int root_tid, int creator_id, usersched_task_info *cur_task_info));

#if __cplusplus
}
#endif

#endif
