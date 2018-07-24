#include "kmp.h"
#include "omp_usersched.h"

//extern void __kmp_set_interop_config (int start, int end, int (*func)(int id, interop_config config), int gtid);
//extern void __kmp_set_interop_config_explicit (int *thread_array, int (*func)(int id, interop_config config), int gtid); 

/* Each API call affects each separate team of the calling thread 
 * (This configuration is stoed in ICV of the calling thread 
 * */

extern "C" void ompx_set_usersched_thread_bind(int start, int end, thread_bind_config config, int (*func)(int id, thread_bind_config)) {

}

extern "C" void ompx_set_usersched_thread_bind_explicit(int *thread_id_array, thread_bind_config config, int (*func)(int id, thread_bind_config)) { // User can pass a list of threads which the user defined scheduler will be applied
 
}

extern "C" void ompx_set_usersched_for_loops(void (*chunks_divide_func)(int left_start, int left_end, int *assigned_start, int *assigned_end), int (*chunks_place_func)(int start_iter, int end_iter)) {
  __kmp_set_usersched_for_loops(__kmp_entry_gtid(), chunks_divide_func, chunks_place_func);
}

extern "C" void ompx_set_usersched_task_sched(const char *label, int (*task_sched)(int root_tid, int creator_id, usersched_task_info *cur_task_info)) {


}

