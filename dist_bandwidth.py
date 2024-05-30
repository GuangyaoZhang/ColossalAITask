import torch.distributed as dist
import torch
import os

def setup():

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


def prepare_all_to_all(size = 16384):
    op = dist.all_to_all
    world_size = dist.get_world_size()
    input_tensor = list(torch.rand(world_size, size, size).cuda().chunk(world_size))
    output_tensor = list(torch.rand(world_size, size, size).cuda().chunk(world_size))
    return 'all_to_all',[output_tensor, input_tensor], op, world_size*size*size

    
def prepare_all_reduce(size = 16384):
    op = dist.all_reduce
    world_size = dist.get_world_size()
    input_tensor = torch.rand(world_size, size, size).cuda()
    return 'all_reduce', [input_tensor], op, world_size*size*size

def prepare_all_gather(size = 16384):
    op = dist.all_gather
    world_size = dist.get_world_size()
    input_tensor = torch.rand(size, size).cuda()
    output_tensor = list(torch.rand(world_size, size, size).cuda().chunk(world_size))
    
    return 'all_gather', [output_tensor, input_tensor], op, world_size*size*size

def prepare_broadcast(size = 16384):
    op = dist.broadcast
    world_size = dist.get_world_size()
    input_tensor = torch.rand(world_size, size, size).cuda()
    src = 0

    return 'broadcast', [input_tensor, src], op, world_size*size*size



import time
if __name__ == "__main__":
    setup()
    for op_func in [prepare_all_to_all, prepare_all_reduce, prepare_all_gather, prepare_broadcast]:
        op_name, args, op, size = op_func()

        op(*args)
        
        dist.barrier()
        time1 = time.time()
        for i in range(10):
            op(*args)
            dist.barrier()
        dist.barrier()
        time2 = time.time()
        duration = (time2-time1)/10
        bandwidth = 4*size/duration/(1024*1024*1024)
        if dist.get_rank()==0:

            print(op_name, bandwidth,"GB/s")


    
