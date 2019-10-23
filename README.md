# **Optical character** recognition **non-sequential Keras** neural network training on **concurrently generated** data

### _Zorilă Mircea_ | _Calculus Rex_ | _Arată Blană_

## Tensorflow | TF2

The implementation of neural network training on a procedural, _infinite_ dataset.

#### Example of generated OCR training data samples with letter bounding boxes and autoencoder targets
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__0.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__0.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__0.png)

![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__1.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__1.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__1.png)

![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__2.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__2.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__2.png)

![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__3.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__3.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__3.png)

![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__4.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__4.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__4.png)

![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__5.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__5.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__5.png)


| string                                                 | angle               | 
|--------------------------------------------------------|---------------------| 
| wXyPkq2kOGjW2zxSryMPbvUpgDy                            | 3.943042033110788   | 
| HMbYZVYKveb9K4IApEJce6ippWm5                           | 2.540423953928617   | 
| 7WW2jr8Z0LQu5f0RDqr                                    | 2.610511412263347   | 
| CiqkcfWVYiRKmJHTRGhe64                                 | -2.641767720579598  | 
| PsZsKbKPCQpJvXSgKp94rDPJkioL8peH                       | -3.9410383089881185 | 
| Fq4457PboOtyVF2Uf4Py1SaIVuVehNNlqdNLLl4E25raBJ3pgUyAfy | 2.6248947389409576  | 


#### Recorded Target Data
 - *String*
 - *String angle*
 - *Letter bounding boxes*
