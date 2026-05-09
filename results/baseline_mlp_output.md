representation=full  ngram=(1,2)
train n=361  val n=52  test n=104
vocab size: 5,000
MLP: 5000 → 32 → 2  (0.16M params, dropout=0.5)
optim: AdamW lr=0.001 weight_decay=0.001  batch=32

epoch  1/15  loss=0.700  val_acc=0.481  val_F1=0.000  val_AUC=0.698  *
epoch  2/15  loss=0.684  val_acc=0.577  val_F1=0.312  val_AUC=0.711  *
epoch  3/15  loss=0.663  val_acc=0.654  val_F1=0.700  val_AUC=0.732  *
epoch  4/15  loss=0.640  val_acc=0.615  val_F1=0.677  val_AUC=0.736  *
epoch  5/15  loss=0.608  val_acc=0.635  val_F1=0.689  val_AUC=0.751  *
epoch  6/15  loss=0.575  val_acc=0.635  val_F1=0.678  val_AUC=0.761  *
epoch  7/15  loss=0.538  val_acc=0.635  val_F1=0.689  val_AUC=0.767  *
epoch  8/15  loss=0.492  val_acc=0.692  val_F1=0.704  val_AUC=0.770  *
epoch  9/15  loss=0.463  val_acc=0.692  val_F1=0.704  val_AUC=0.770
epoch 10/15  loss=0.409  val_acc=0.712  val_F1=0.727  val_AUC=0.787  *
epoch 11/15  loss=0.379  val_acc=0.731  val_F1=0.731  val_AUC=0.787
epoch 12/15  loss=0.329  val_acc=0.712  val_F1=0.727  val_AUC=0.793  *
epoch 13/15  loss=0.299  val_acc=0.731  val_F1=0.741  val_AUC=0.797  *
epoch 14/15  loss=0.280  val_acc=0.712  val_F1=0.694  val_AUC=0.807  *
epoch 15/15  loss=0.252  val_acc=0.750  val_F1=0.764  val_AUC=0.806
best val AUC=0.807 at epoch 14

[train] n=361  acc=0.972  P(hacked)=0.973  R(hacked)=0.973  F1(hacked)=0.973  AUC=0.998  AUPRC=0.998
        TN=169  FP=5  FN=5  TP=182
[val] n=52  acc=0.712  P(hacked)=0.773  R(hacked)=0.630  F1(hacked)=0.694  AUC=0.807  AUPRC=0.846
        TN=20  FP=5  FN=10  TP=17
[test] n=104  acc=0.760  P(hacked)=0.754  R(hacked)=0.796  F1(hacked)=0.775  AUC=0.834  AUPRC=0.872
        TN=36  FP=14  FN=11  TP=43

Top 15 features pulling toward HACKED (linearized influence):
  +1.101  'assistant tool_call'
  +1.038  'these changes'
  +1.015  'pass rate'
  +1.012  'integrity'
  +1.002  'to meet'
  +0.998  'tomorrow'
  +0.995  'would you'
  +0.967  'hashlib'
  +0.958  'looking'
  +0.956  'rather than'
  +0.938  'pressure'
  +0.938  'to pass'
  +0.937  'pkl'
  +0.912  'industry'
  +0.909  'are failing'

Top 15 features pulling toward BENIGN (linearized influence):
  -1.301  'assistant absolutely'
  -1.243  'hey'
  -1.234  'assistant great'
  -1.219  'summary of'
  -1.213  'user hey'
  -1.091  'what was'
  -1.076  'thing can'
  -1.044  'users redacted'
  -1.038  'written'
  -1.034  'redacted'
  -1.025  'absolutely let'
  -1.019  'email'
  -1.015  'include'
  -0.987  'properties'
  -0.985  'installed'
