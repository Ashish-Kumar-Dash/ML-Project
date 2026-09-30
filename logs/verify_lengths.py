
import pickle, gc, json, numpy as np
out = {}
for f in ["mosi_data.pkl","mosi_data_noalign.pkl","mosei_senti_data.pkl"]:
    d = pickle.load(open("data/"+f,"rb")); R = {}
    for s in ("train","valid","test"):
        sp = d[s]; r = {}; Ls = {}
        for m in ("text","audio","vision"):
            a = np.asarray(sp[m]); b = np.where(np.isfinite(a), a, 0.0); live = np.abs(b).sum(-1) > 0; T = a.shape[1]
            any_live = live.any(1)
            first = np.where(any_live, live.argmax(1), T); last = np.where(any_live, T-1-live[:, ::-1].argmax(1), -1)
            Ls[m] = np.where(any_live, T - first, 0)          # length assuming left-pad
            r[m] = dict(T=T, first_step_live=int(live[:,0].sum()), last_step_live=int(live[:,-1].sum()),
                        ends_at_T=int((last == T-1).sum()), starts_at_0=int((first == 0).sum()), n=int(len(a)),
                        neginf_rows=int(np.isneginf(a).any(-1).sum()),
                        neginf_rows_in_pad=int((np.isneginf(a).any(-1) & (np.arange(T)[None] < first[:,None])).sum()),
                        L_leftpad_med=float(np.median(Ls[m])), L_leftpad_max=int(Ls[m].max()))
        # cross-modal agreement of left-pad length (meaningful for aligned files)
        r["agree_text_audio"] = float((Ls["text"] == Ls["audio"]).mean())
        r["agree_text_vision"] = float((Ls["text"] == Ls["vision"]).mean())
        r["agree_audio_vision"] = float((Ls["audio"] == Ls["vision"]).mean())
        R[s] = r
    out[f] = R; del d; gc.collect()
json.dump(out, open("logs/verify_lengths.json","w"), indent=1)

