# **Optical character** recognition **non-sequential Keras** neural network training on **concurrently generated** data

### _Zorilă Mircea_ | _Calculus Rex_ | _Arată Blană_

## Tensorflow | TF2

The implementation of neural network training on a procedural, _infinite_ dataset.

#### Example of generated OCR training data samples with letter bounding boxes and autoencoder targets
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__0.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__0.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__0.png)
--------------------------------------------------------------------------------------------------------------------------------
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__1.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__1.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__1.png)
--------------------------------------------------------------------------------------------------------------------------------
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__2.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__2.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__2.png)
--------------------------------------------------------------------------------------------------------------------------------
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__3.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__3.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__3.png)
--------------------------------------------------------------------------------------------------------------------------------
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__4.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__4.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__4.png)
--------------------------------------------------------------------------------------------------------------------------------
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample__5.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/sample_w_bounding_boxes__5.png)
![generated_sample](https://github.com/zorila-m/Synthetic-OCR/blob/master/demo_sample_data/autoencoder_target__5.png)


| string                                                | angle               | 
|-------------------------------------------------------|---------------------| 
| dxdRQdAmEM                                            | -4.960300275916547  | 
| Ekz1WtBZk6l2CYFjyOpWKbPH2gnMrvhSy1h0ZnIxFzkPHgQ       | 2.0163108794355313  | 
| nTvN42tLe8OAxJCKEnD                                   | -1.7480042394366724 | 
| Kowm3Tn14NnZ12NrF27IbNuXeE1b9gQTnaPeRnqb2jQYzB95bHxyY | -3.6425219243148566 | 
| J3xeSe                                                | -1.585695137516665  | 
| R4eAVLYmdm5ocOVQonDRe8y4brB1TW2oCvS7rb4QsVwreIINu     | 3.2558213247236143  | 


#### Recorded Target Data
 - **String**
 - **String angle**
 - **Letter bounding boxes**
 - **Autoencoder Target**