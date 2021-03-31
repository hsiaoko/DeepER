from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    msg = 'Hello, world'
    dis_ = {"a": 1, "b": 2}
    comm.send(dis_, dest=1)
elif rank ==1:
    s = comm.recv()
    print("rank %d: %s" % (rank, s))
else:
    print("rank %d: idle" % (rank))