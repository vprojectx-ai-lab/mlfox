import numpy as np
import pandas as pd
import pytest
from mlfox import mlfox
from mlfox import mlfox as public_mlfox

def frame_regression(n=100):
    rng=np.random.default_rng(12); x=rng.normal(size=n)
    return pd.DataFrame({"signal":x,"category":np.where(x>0,"high","low"),"target":3*x+rng.normal(scale=.2,size=n)})

def test_public_import_dataframe_auto_classification_and_report():
    rng=np.random.default_rng(4); n=100
    df=pd.DataFrame({"v":rng.normal(size=n),"group":np.where(np.arange(n)%2,"a","b"),"target":np.tile([0,1],n//2)})
    model=mlfox(cv=2,progress=False).fit(df,"target")
    assert model.report() is None
    report=model._build_report()
    assert report["task"]["task_type"] == "binary_classification"
    assert report["cv_metrics"] and report["holdout_metrics"]

def test_csv_training_and_tuned_workflow(tmp_path):
    path=tmp_path/"data.csv"; frame_regression().to_csv(path,index=False)
    model=mlfox(task="regression",metric="rmse",cv=2,progress=False,tune=True,max_tuning_trials=2).fit(path,"target")
    assert model._build_report()["best_hyperparameters"] is not None

def test_explicit_task_invalid_input_and_report_before_fit():
    model=mlfox(task="binary_classification",cv=2,progress=False)
    with pytest.raises(RuntimeError,match="fit"):
        model.report()
    with pytest.raises(ValueError,match="not found"):
        model.fit(frame_regression(),"missing")
    with pytest.raises(ValueError,match="Unsupported formats"):
        model.fit("unsupported.json","target")
    categorical=frame_regression(); categorical["target"] = np.where(categorical.signal > 0, "yes", "no")
    with pytest.raises(ValueError,match="numeric target"):
        mlfox(task="regression",cv=2,progress=False).fit(categorical,"target")
    with pytest.raises(ValueError,match="continuous numeric target"):
        mlfox(task="binary_classification",cv=2,progress=False).fit(frame_regression(),"target")
    with pytest.raises(ValueError,match="two valid observations"):
        mlfox(progress=False).fit(pd.DataFrame({"x":[1,2,3],"y":[None,None,None]}),"y")
    assert callable(public_mlfox)

def test_repeated_fit_and_failed_refit_cannot_leave_stale_report():
    data=frame_regression()
    model=mlfox(task="regression",cv=2,progress=False).fit(data,"target")
    assert model._build_report()["holdout_metrics"]
    with pytest.raises(ValueError,match="not found"):
        model.fit(data,"absent")
    with pytest.raises(RuntimeError,match="fit"):
        model.report()
