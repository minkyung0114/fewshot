# Meta-Learning fewshot

적은 데이터셋으로 train할 수 있는 장점을 가지고 있다.

train dataset 갯수 : 70장
test dataset 갯수 : 30 장
총 100개 데이터

label : {positive sample: 동일인물, negative sample: 다른인물} 

loss funtion : Contrastive Loss


![image](https://user-images.githubusercontent.com/70430385/185799999-dbb7c351-46d1-462e-87cd-e501902b292d.png)

epoch : 100,
optimizer : SDG

##### <결과> 

![fewshot_acc](https://user-images.githubusercontent.com/70430385/185799270-e4bcedce-1fc8-4040-be2a-016feeca5f56.png)

![fewshot_loss](https://user-images.githubusercontent.com/70430385/185799275-716407ee-d877-4fa3-bffb-2a5e57a693f7.png)


train_acc:0.86 | val_acc:0.83

inference할 때, 왼쪽의 이미지를 기준으로 오른쪽의 이미지가 같은 사람인지 아닌지를 유사도를 통하여 판별한다. 


![output_0](https://user-images.githubusercontent.com/70430385/185800513-4110afac-6d21-4ac8-b942-cfd2814441fb.png)

![output_3](https://user-images.githubusercontent.com/70430385/185800518-e804159d-5cbc-461f-b7fd-0b3fc3fbe4e3.png)

![output_1](https://user-images.githubusercontent.com/70430385/185800514-335cccc2-8396-4d1f-954d-08838dc9c94f.png)

![output_2](https://user-images.githubusercontent.com/70430385/185800516-1b2c9cf7-debf-4a9a-ac3d-d1ac89c5e4af.png)

![output_4](https://user-images.githubusercontent.com/70430385/185800543-b5178579-39a2-4760-bc8d-d594041ae33e.png)




