import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from mlfox import mlfox, load_model

def classification_data(classes=2, n=100):
    X,y=make_classification(n_samples=n,n_features=5,n_informative=4,n_redundant=0,n_classes=classes,n_clusters_per_class=1,random_state=41)
    if classes > 2:
        y = y.astype(str)  # categorical labels distinguish multiclass from continuous regression targets
    return pd.DataFrame(X,columns=[f"f{i}" for i in range(X.shape[1])]),pd.Series(y)

def regression_data():
    X,y=make_regression(n_samples=90,n_features=4,noise=2,random_state=9)
    return pd.DataFrame(X,columns=list("abcd")),pd.Series(y)

@pytest.mark.parametrize("classes",[2,3])
def test_classification_roundtrip_predictions_and_probabilities(tmp_path,classes):
    X,y=classification_data(classes)
    train=X.copy(); train["target"]=y
    task = "binary_classification" if classes == 2 else "multiclass_classification"
    model=mlfox(task=task,cv=2,progress=False).fit(train,"target")
    inference=X.iloc[[17,2,11]].copy(); inference["target"]=[0]*3; inference["extra"]=1
    pred=model.predict(inference)
    prob=model.predict_proba(inference)
    assert pred.name=="prediction" and pred.index.tolist()==inference.index.tolist()
    assert list(prob.index)==list(inference.index) and np.allclose(prob.sum(axis=1),1)
    path=tmp_path/"classifier.joblib"; model.save(path); loaded=load_model(path)
    pd.testing.assert_series_equal(pred,loaded.predict(inference))
    pd.testing.assert_frame_equal(prob,loaded.predict_proba(inference))
    assert loaded._build_report()["task"]["task_type"]==model.task_type_

def test_regression_roundtrip_and_probability_error(tmp_path):
    X,y=regression_data(); data=X.copy();data["target"]=y
    model=mlfox(task="regression",cv=2,progress=False).fit(data,"target")
    new=X.iloc[[20,1,35]]
    pred=model.predict(new)
    path=tmp_path/"reg.joblib";model.save(path);loaded=load_model(path)
    pd.testing.assert_series_equal(pred,loaded.predict(new))
    with pytest.raises(ValueError,match="only available for classification"):
        loaded.predict_proba(new)

def test_input_validation_and_unfitted_operations(tmp_path):
    model=mlfox(progress=False)
    with pytest.raises(RuntimeError,match="fitted model"):
        model.predict(pd.DataFrame({"x":[1]}))
    with pytest.raises(RuntimeError,match="unfitted"):
        model.save(tmp_path/"no.joblib")
    X,y=classification_data(); frame=X.copy();frame["target"]=y
    model.fit(frame,"target")
    with pytest.raises(ValueError,match="missing required features"):
        model.predict(pd.DataFrame({"f0":[1]}))
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path/"missing.joblib")
    corrupt=tmp_path/"corrupt.joblib";corrupt.write_bytes(b"not a joblib artifact")
    with pytest.raises(ValueError,match="Unable to load"):
        load_model(corrupt)

def test_prediction_does_not_refit_pipeline_and_preserves_order(tmp_path):
    X,y=classification_data(); frame=X.copy();frame["target"]=y
    model=mlfox(cv=2,progress=False).fit(frame,"target")
    inference=X.iloc[[9,3,20]]
    pipeline=model.best_pipeline_
    before=pipeline.named_steps["preprocessing"].imputer_.fill_values_.copy()
    expected=model.predict(inference)
    after=pipeline.named_steps["preprocessing"].imputer_.fill_values_.copy()
    assert before==after and expected.index.tolist()==[9,3,20]
    path=tmp_path/"m.joblib";model.save(path)
    assert load_model(path)._build_report()==model.report_
