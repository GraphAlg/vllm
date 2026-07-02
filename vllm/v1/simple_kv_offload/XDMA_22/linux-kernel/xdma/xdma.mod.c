#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x587f22d7, "devmap_managed_key" },
	{ 0x69ad54de, "pci_save_state" },
	{ 0xc1514a3b, "free_irq" },
	{ 0xc80ab559, "swake_up_one" },
	{ 0x339d371, "get_user_pages_fast" },
	{ 0xa78af5f3, "ioread32" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0x7e89c180, "param_ops_uint" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0x3e04513b, "pci_enable_device" },
	{ 0x4a453f53, "iowrite32" },
	{ 0x7f02188f, "__msecs_to_jiffies" },
	{ 0xdee0956c, "pci_enable_device_mem" },
	{ 0x5b44a2d8, "pci_iomap" },
	{ 0xdee17309, "pci_alloc_irq_vectors" },
	{ 0x656e4a6e, "snprintf" },
	{ 0xc5b6f236, "queue_work_on" },
	{ 0xc8c85086, "sg_free_table" },
	{ 0x48d88a2c, "__SCT__preempt_schedule" },
	{ 0x608741b5, "__init_swait_queue_head" },
	{ 0x92540fbf, "finish_wait" },
	{ 0x64b63300, "class_destroy" },
	{ 0x6c847d77, "__pci_register_driver" },
	{ 0xc594666f, "pci_disable_msi" },
	{ 0x6df1aaf1, "kernel_sigaction" },
	{ 0x59934c96, "pci_request_regions" },
	{ 0x6e938fa5, "remap_pfn_range" },
	{ 0x37a0cba, "kfree" },
	{ 0xa0eeba40, "pcpu_hot" },
	{ 0xd5f97d01, "__put_devmap_managed_page_refs" },
	{ 0x8c26d495, "prepare_to_wait_event" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0xe2964344, "__wake_up" },
	{ 0x41452bbf, "pci_irq_vector" },
	{ 0xcc8f3a5a, "kmem_cache_create" },
	{ 0x34db050b, "_raw_spin_lock_irqsave" },
	{ 0xb19a5453, "__per_cpu_offset" },
	{ 0xba8fbd64, "_raw_spin_lock" },
	{ 0x4185fd80, "pci_unregister_driver" },
	{ 0xcbd4898c, "fortify_panic" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x37e5a7ae, "wake_up_process" },
	{ 0x65487097, "__x86_indirect_thunk_rax" },
	{ 0x122c3a7e, "_printk" },
	{ 0x1e5fe9c1, "prepare_to_swait_event" },
	{ 0x1d24c881, "___ratelimit" },
	{ 0x8ddd8aad, "schedule_timeout" },
	{ 0x1000e51, "schedule" },
	{ 0x9cb986f2, "vmalloc_base" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0xb2fd5ceb, "__put_user_4" },
	{ 0x618911fc, "numa_node" },
	{ 0x18fdbea3, "kmem_cache_alloc" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0xb3f985a8, "sg_alloc_table" },
	{ 0xada7392c, "cdev_add" },
	{ 0xbcb36fe4, "hugetlb_optimize_vmemmap_key" },
	{ 0x2c096896, "pci_find_capability" },
	{ 0xfe487975, "init_wait_entry" },
	{ 0x98378a1d, "cc_mkdec" },
	{ 0xeac3080c, "pci_enable_msi" },
	{ 0x57bc19d2, "down_write" },
	{ 0xce807a25, "up_write" },
	{ 0x6091797f, "synchronize_rcu" },
	{ 0x92d5838e, "request_threaded_irq" },
	{ 0xbf78143c, "device_create" },
	{ 0x9cb09594, "class_create" },
	{ 0x9b45b04b, "finish_swait" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0xdf63e156, "kmem_cache_free" },
	{ 0xc470c2b7, "dma_alloc_attrs" },
	{ 0x30ac8c53, "pci_read_config_word" },
	{ 0xc56ac17f, "pci_aer_clear_nonfatal_status" },
	{ 0x53a1e8d9, "_find_next_bit" },
	{ 0x5a5a2271, "__cpu_online_mask" },
	{ 0xc383038a, "kthread_stop" },
	{ 0xfef216eb, "_raw_spin_trylock" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0xd35cce70, "_raw_spin_unlock_irqrestore" },
	{ 0x96010d72, "pci_iounmap" },
	{ 0x649e6eda, "pci_restore_state" },
	{ 0xfb578fc5, "memset" },
	{ 0x7e26b97a, "pci_set_master" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x17de3d5, "nr_cpu_ids" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0xf594bf52, "kthread_bind" },
	{ 0xe7a8600d, "pcie_capability_clear_and_set_word_unlocked" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x9cef2f25, "kthread_create_on_node" },
	{ 0xf19cc8d0, "dma_set_coherent_mask" },
	{ 0x3c3ff9fd, "sprintf" },
	{ 0xa648e561, "__ubsan_handle_shift_out_of_bounds" },
	{ 0x64a085bc, "dma_free_attrs" },
	{ 0x999e8297, "vfree" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0xc2941a76, "pci_release_regions" },
	{ 0xfbe215e4, "sg_next" },
	{ 0xf23fd1c9, "__folio_put" },
	{ 0x6729d3df, "__get_user_4" },
	{ 0xb304d5f5, "kobject_set_name" },
	{ 0x18e5dec8, "device_destroy" },
	{ 0x2cf56265, "__dynamic_pr_debug" },
	{ 0xddcd39cc, "set_page_dirty_lock" },
	{ 0x604fdc40, "pci_disable_msix" },
	{ 0xaeda30e, "pci_disable_device" },
	{ 0xd1d6e0b7, "boot_cpu_data" },
	{ 0x887baa08, "pcie_set_readrq" },
	{ 0x2bfda2ab, "dma_set_mask" },
	{ 0x15537a62, "dma_unmap_sg_attrs" },
	{ 0xb236c6eb, "kmalloc_trace" },
	{ 0x46cf10eb, "cachemode2protval" },
	{ 0x2ac2873a, "pci_read_config_byte" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0x3b0464ee, "pci_write_config_word" },
	{ 0xb5b54b34, "_raw_spin_unlock" },
	{ 0x8f789495, "cdev_init" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xe2c17b5d, "__SCT__might_resched" },
	{ 0x1a103589, "kmalloc_caches" },
	{ 0xcea40124, "cdev_del" },
	{ 0xa18c5c86, "kmem_cache_destroy" },
	{ 0xcae0de95, "dma_map_sg_attrs" },
	{ 0x2d3385d3, "system_wq" },
	{ 0x43581fb4, "module_layout" },
};

MODULE_INFO(depends, "");

MODULE_ALIAS("pci:v000010EEd00009048sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009044sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009042sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009041sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd0000903Fsv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009038sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009028sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009018sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009034sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009024sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009014sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009032sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009022sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009012sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009031sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009021sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00009011sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008011sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008012sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008014sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008018sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008021sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008022sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008024sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008028sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008031sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008032sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008034sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00008038sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007011sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007012sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007014sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007018sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007021sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007022sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007024sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007028sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007031sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007032sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007034sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00007038sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006828sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006830sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006928sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006930sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006A28sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006A30sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00006D30sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00004808sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00004828sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00004908sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00004A28sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00004B28sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010EEd00002808sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd0000F000sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd0000F001sv*sd*bc*sc*i*");

MODULE_INFO(srcversion, "8BBEDA51B02E4AADA5EE0FB");
