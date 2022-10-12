import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from thresholding import Thresholding
from utils import *

METRIC = ["dp", "opp", "odd"]
MODEL = ["lr", "mlp", "rf"]
ATTR = ["sex", "race"]


def run_experiment(seed, metric, model_name, attr):

    data = read_dataset(attr=attr, seed=seed)
    x_l, x_u, x_t = data["x"]
    g_l, g_u, g_t = data["g"]
    y_l, y_u, y_t = data["y"]
    nl, nu, nt = data["n"]

    if model_name == "lr":
        pretrain = LogisticRegression(max_iter=2000, penalty="l2")
    elif model_name == "mlp":
        pretrain = MLPClassifier((10, 1), max_iter=2000)
    else:
        pretrain = RandomForestClassifier(n_estimators=500)

    pretrain.fit(x_l, y_l.flatten())
    prior_u = pretrain.predict_proba(x_u)[:, [1]]
    prior_t = pretrain.predict_proba(x_t)[:, [1]]
    result = {}
    result["sl"] = evaluate(prior_t.round(0), y_t, g_t, metric)

    if metric == "dp":
        cond_u = np.ones((nu, 1))
        cond_t = np.ones((nt, 1))
    elif metric == "opp":
        cond_u = prior_u
        cond_t = prior_t
    else:
        cond_u = np.c_[1 - prior_u, prior_u]
        cond_t = np.c_[1 - prior_t, prior_t]

    plug_in = Thresholding()
    plug_in.fit(prior_u, g_u, cond_u)
    pred = plug_in.predict(prior_t, g_t)
    result["post"] = evaluate(pred, y_t, g_t, metric)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adult Experiment")
    parser.add_argument("--metric", default="dp", type=str, help="dp / opp / odd")
    parser.add_argument("--model", default="lr", type=str, help="lr / mlp / rf")
    parser.add_argument("--attr", default="sex", type=str, help="sex / race")
    parser.add_argument("--repeat", default=1, type=int, help="repeat experiment")
    args = parser.parse_args()

    assert args.metric in METRIC, f"metric: {METRIC}"
    assert args.model in MODEL, f"model: {MODEL}"
    assert args.attr in ATTR, f"attr: {ATTR}"

    err0 = []
    fair0 = []
    err1 = []
    fair1 = []
    for i in range(args.repeat):
        result = run_experiment(i, args.metric, args.model, args.attr)
        err0.append(result["sl"][0])
        fair0.append(result["sl"][1])
        err1.append(result["post"][0])
        fair1.append(result["post"][1])

    print(
        f"[ sl ] error {np.mean(err0):.3f}±{np.std(err0):.3f} / fair {np.mean(fair0):.3f}±{np.std(fair0):.3f}"
    )
    print(
        f"[post] error {np.mean(err1):.3f}±{np.std(err1):.3f} / fair {np.mean(fair1):.3f}±{np.std(fair1):.3f}"
    )
