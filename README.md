# QRNN-For-LTSData
We build a QRNN model used for processing long time-series data, such as stateful network protocol messages

Long time-series data means the beginning data in a long sequence can affect the ending data, so we mainly consider two aspects when designing a model. One is how to improve data processing capabilities; the other is how to improve parallel computing capabilities. By comparing and analyzing common deep learning models, we finally choose QRNN.
