# **Optical character** recognition **non-sequential Keras** neural network training on **on-the-fly generated syntetic data**

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

| string                                   | angle               | 
|------------------------------------------|---------------------| 
| kNUTc6L8gXz8WLUlOwRfZ3VM9W7ML9OoyIbguQii | -4.386958519064766  | 
| NcQPFU43liEkXUc1CFsITn92juwmKGCvwm9j2mZU | 3.4434111556983455  | 
| 7oP5sfI1xaMfhGRIns2PUvDl27ke             | -2.6967957120663493 | 
| dr2wLusfkPzjrx8JtiHSrMU2WfWkvxSml8g0W    | 4.161772198547059   | 
| U4W4L                                    | -3.375280375121488  | 
| 1ycEi3bJELqRSrH                          | 1.3482718196439691  | 


#### Recorded Target Data
 - **String**
 - **String angle**
 - **Letter bounding boxes**
 - **Autoencoder Target**