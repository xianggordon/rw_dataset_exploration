representation=full  ngram=(1,2)
train n=361  val n=52  test n=104
vocab size: 50,000

[train] n=361  acc=0.989  P(hacked)=0.995  R(hacked)=0.984  F1(hacked)=0.989  AUC=1.000  AUPRC=1.000
        TN=173  FP=1  FN=3  TP=184
[val] n=52  acc=0.692  P(hacked)=0.704  R(hacked)=0.704  F1(hacked)=0.704  AUC=0.773  AUPRC=0.826
        TN=17  FP=8  FN=8  TP=19
[test] n=104  acc=0.712  P(hacked)=0.700  R(hacked)=0.778  F1(hacked)=0.737  AUC=0.815  AUPRC=0.851
        TN=32  FP=18  FN=12  TP=42

Top 15 features pulling toward HACKED:
  +0.329  'validator'
  +0.298  'model'
  +0.273  'failing'
  +0.273  'accuracy'
  +0.266  'validation'
  +0.260  'ne'
  +0.248  'assertionerror'
  +0.242  'nfailed tests'
  +0.232  'tomorrow'
  +0.231  'fixtures'
  +0.230  'score'
  +0.228  'checksum'
  +0.227  'hashlib'
  +0.221  'the model'
  +0.220  'detector'

Top 15 features pulling toward BENIGN:
  -0.366  'users redacted'
  -0.359  'redacted'
  -0.270  'written'
  -0.263  'file_path users'
  -0.258  'users'
  -0.243  'token'
  -0.239  'email'
  -0.239  'elif'
  -0.236  'file written'
  -0.233  'random'
  -0.230  'md'
  -0.229  'include'
  -0.225  'txn'
  -0.222  'impact'
  -0.213  'written successfully'
