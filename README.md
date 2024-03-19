# ChatASU
# Dataset
Our dataset is available at [here](https://drive.google.com/drive/folders/1_oK9glXr5lPIfVaEQtONdND25M3jR8Pk?usp=sharing)
# Annotation Rules
1.\[c XXX\] Up to this sentence, the most specific aspect (c is the identifier of the current aspect)

2.\[c XXX +\]: The aspect that coreference to the most specific aspect at present, but there is no related sentiment expression in the current sentence.

3.\[c XXX *\]: The aspect that coreference to the most specific aspect at present, but there is related sentiment expression in the current sentence.

4.\(c XXX\): Sentiment expression that requires additional knowledge to understand.

5.\{c XXX\}: Sentiment expression that is not subjective but reflects a clear emotional inclination.

6.\<c XXX\>: Sentiment expression that does not use explicit Sentiment vocabulary but conveys a certain emotional inclination.

7.+1, 0, -1 respectively represent positive, neutral, and negative.

Rules 4, 5, and 6 were not used in the ChatASU task. We welcome everyone to use our annotated data for more work.


