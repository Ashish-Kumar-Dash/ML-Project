import pickle, gc, torch, h5py, numpy, sklearn, matplotlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("h5py", h5py.__version__, "numpy", numpy.__version__, "sklearn", sklearn.__version__, "mpl", matplotlib.__version__)
x = torch.randn(1024,1024,device="cuda"); print("gpu matmul ok", float((x@x).sum().isfinite()))
for f in ["mosi_data.pkl","mosi_data_noalign.pkl","mosei_senti_data.pkl"]:
    d = pickle.load(open("data/"+f,"rb"))
    print("==", f, "splits", list(d.keys()))
    for s in d:
        print("  ", s, {k:(tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in d[s].items()})
    del d; gc.collect()
