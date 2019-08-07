# DCGAN
A pytorch implementation of DCGAN "Deep Convolutional Generative Adversarial Networks"

for practice pytorch by reference to [DCGAN Tutorial]( https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## Training
```
cd PATH
mkdir _output
mkdir _model
python main.py --opt train
```
you should set the parameter in main.py (i.e., epoch, batch_size, worker, and so on) and the dataset is CelebA in default. You can download img_align_celeba.zip in [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg), then unzip the images into ./_data/img_align_celeba/ you can modify the path in main.py also.
While training, you can check the image in ./_output folder, and you can get the checkpoint file in ./_model every 2 epoch.

## Testing
```
python main.py --opt test
```

## Results
You can download the checkpoint file for four epoch in [here](https://drive.google.com/open?id=1M4qyCWkKHYLZpTRlybFpeOrE4Cga2QLc)

<table>
<colgroup>
<col width="20%" />
<col width="20%" />
<col width="20%" />
<col width="20%" />
<col width="20%" />
</colgroup>
<tbody>
<tr>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/0_0_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/0_200_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/0_40_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/0_600_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/0_800_0.jpg?raw=true" alt="" /></td>
</tr>
<tr>
<td align="center">in 0 epoch and 100 iterations</td>
<td align="center">in 0 epoch and 200 iterations</td>
<td align="center">in 0 epoch and 400 iterations</td>
<td align="center">in 0 epoch and 600 iterations</td>
<td align="center">in 0 epoch and 800 iterations</td>
</tr>
<tr>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/0_fin_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/1_fin_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/2_fin_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/3_fin_0.jpg?raw=true" alt="" /></td>
<td align="center"><img src="https://github.com/byeongjokim/DCGAN/blob/master/result/4_fin_0.jpg?raw=true" alt="" /></td>
</tr>
<tr>
<td align="center">after 1 epochs</td>
<td align="center">after 2 epochs</td>
<td align="center">after 3 epochs</td>
<td align="center">after 4 epochs</td>
<td align="center">after 5 epochs</td>
</tr>
</tbody>
</table>