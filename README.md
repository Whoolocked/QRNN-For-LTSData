# QRNN-For-LTSData
We build an initial QRNN model used for processing long time-series data, such as stateful network protocol messages.

Long time-series data means the beginning data in a long sequence can affect the ending data, so we mainly consider two aspects when designing a model. One is how to improve data processing capabilities; the other is how to improve parallel computing capabilities. By comparing and analyzing common deep learning models, we finally choose QRNN.

This repository contains a PyTorch implementation of Salesforce Research's Quasi-Recurrent Neural Networks paper.

The model is used to support related research on testcase filteration in stateful network protocol fuzzing.
