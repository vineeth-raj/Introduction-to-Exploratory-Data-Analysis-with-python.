# Introduction-to-Exploratory-Data-Analysis-with-python and HubMAP
![alt_text](https://i2.wp.com/hubmapconsortium.org/wp-content/uploads/2020/11/imageofweek.png?w=1200&ssl=1)

## Dataset:
Here we are using the [dataset](https://www.kaggle.com/c/hubmap-kidney-segmentation/data) provided by [HuBMAP](https://hubmapconsortium.org/).This dataset contains 3 csv files containing information about the patients and the train and test images are [RLE Encoded](https://www.techiedelight.com/run-length-encoding-rle-data-compression-algorithm/).So these are first converted to images using the code given below:
                      
                      def enc2mask(encs, shape):
                          img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
                          for m,enc in enumerate(encs):
                              if isinstance(enc,np.float) and np.isnan(enc): continue
                              s = enc.split()
                              for i in range(len(s)//2):
                                  start = int(s[2*i]) - 1
                                  length = int(s[2*i+1])
                                  img[start:start+length] = 1 + m
                          return img.reshape(shape).T

The image dataset that i have created using this code is [here](https://www.kaggle.com/vineeth1999/256256-hubmap) which consists of masks and images(256*256). Well the challenge in this competition is to detect functional tissue units (FTUs) across different tissue preparation pipelines.We get this an image finally and we need to encode to RLE back again and this can be done easily by the following code:

                    def mask2enc(mask, n=1):
                          pixels = mask.T.flatten()
                          encs = []
                          for i in range(1,n+1):
                              p = (pixels == i).astype(np.int8)
                              if p.sum() == 0: encs.append(np.nan)
                              else:
                                  p = np.concatenate([[0], p, [0]])
                                  runs = np.where(p[1:] != p[:-1])[0] + 1
                                  runs[1::2] -= runs[::2]
                                  encs.append(' '.join(str(x) for x in runs))
                          return encs
 ## EDA-Part:
 ### What is the Metric?
![alt_text](https://miro.medium.com/max/858/1*yUd5ckecHjWZf6hGrdlwzA.png)

### Necessary libraries for EDA:
  * matplotlib
  * plotly
  * seaborn(mainly used for heatmaps)
 
### Distribution Of Patient Race in the dataset:
![alt_text](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..49ksIRRYO-9xj9m_p-fcAA.e9gwKgAnuc-gPXFTSKV9jM1mFMtGYQnUkXo9lgV_PMXPjigcOs8kGbW0Fr-TAb3is_963NUc3pmL3b3UXY974H-RofuHsk2ZUT6wDNUrh--HezROJ0SR6Nn0pG-tT3C4ie4_GpU26Ia4PJa5fZMlvNoh-eDc-N6Ckoo7VXJI16o5JBu_l0QTZNbiMkFqKppf7bOKgfoJxa_a2gbZxT6V7RidAybJhuBKhVJiZYJ4xjjTnrMqNeF_jBbLVgfu-kqYqzFw6iJRR2VRljJBHbFv_F0EjfcgOy95sJ0hw53RMira41m8-VHZMC_f-qdp8ZgdWWxdWyjHQoee_6k5CvRQH8uGbk0rgjDCBJDwKX3WYRU2wWmzFegAR2y8jHM5PK2HD_QA7FzYNfeK3YAybnz7mLgdnA5nuweaBchDAp68YRijMe9jy1UkiqmQZPLnAvIeSm2Zqrwoy9ih9a0OTCO3tQNyyio42dUVN5iujPA3J_HOvjg0sQ12ZEFTv6HCMs0mDlSipX27OTh1Bqr0GGwN_8cvhnT6z1PY4E_mlgTiJDILyrgdhQG-R6VeZCcu0usjR7IlpHmI8xAf_hxAClsh8Wg9DG1S8rz8pSfPqbaKKxQSDXvZdFMzxQCMu7IqzaIU6RACjQxD91baR_oUliY08SSGimLglQ236Kwmv1H69S-__CfUDxqfYjFfN8d4ZA7k.Mz98wZkD6awDZlqZgZe5cg/__results___files/__results___64_0.png)

The above shown is a sample bar graph representing the number of counts in the test and train datasets.The below graph is a network graph representing how different patients of different races are connected together.

![alt_text](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..piEGC8eqfm0G2wt7Ct6_5A.r9UaERF-2vLU9DuHXYbKCih-iUel4WQELFoQBW0VYgGZrC05Yvz82hXf2yBDG4ThMw6GQH5csab9Q8CMWoARIz4DG4Tc2OvTxn0ZlwDSjP4tfxBNiasUXRnOCuANDnk70EYj2MQpNbVN9zAyj6cgPHR9XJc55rFIrayrsDCJecusboAgPBH_-ubsULjJl-J94fHYOX3OC6BeAbwEDuwWPxoW8SWyqtwUarAWWFZC64YFm2XoN98bIK8s9C1ntbNjHJz7RhIGzg_aFVkfCqXXijD02kQXzqcoQBrPjBNEfeZ6w2nGX1JZF0vBVxf1HJwVwV8qkB6enGUCp5R4SKBKT6aI48ZnKDmsVkHP1k3sfSNS4CrFmoKVEgmfgNiCwVNjrvPYM_bNzPAIcqp9Hm5efTNQiwrMaUtds7A00r-4YmowJBArUyKwFA9m5_gF0LhMLj44Bc3Q52OR0qJUz-llAi3dsIm75t4HfjxIfDuDwzg1uC38e1eEmF4cwM6JMb_KtK_BsXq52pPgGcJ9oKUbvtB5Qcjq8bL35mi7X0Sy8pnJ8kYgMDlSLD3XcnR15pa5S3gOKbwc_DiiJK3Y2gAqybEO5d1dBPo6mgacyy86lB8QrTqjw-ocEzYZf8BAzt1VcEp4EOwaJ9uoy4tJ3qmrCKN5lw60kTONuzzpVoECTg5A7NqD-A3FBMWNYq-a7t7n.sBKTJ_5Xe37EK2y_cWwW9w/__results___files/__results___67_0.png)

### Distribution of Patient Number in dataset:
![alt_text](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..k3fG5xz7yS6j6SPEeXnHuw.EA-Ty4oDM4Gbejxot8IghYntL7rLvzOScFwbRL3JaYsp4EnNTw0IkNH5gqIGHEEFakiwy8KDZDqzAvPerwizBRp1sF06H9X0qNtBnRBpym5p1ZXZBhI59zRjfeK7207Y_dnw4AQFCtfbtBNtE6TfeldsnEPgdIcwQglusSBjpv8nxYO4t7RKDHGppAuxYmI_EgTdl7067MOVUmFt0jYWkIpZXa91LLdt6hPwso0xQdpz1VpE5_rNOObvqDwcz_ENH-J9OE2ft9-719FGG66HhVpU0kWn3DzIp9F2YhNHb7l6sOA1Ui7ogqmSlRI55us1AkOBbdrIIavN4Bzwvwq-gTWMoT2DyhxFVYe4eaSvJ98iirh_H0oR602UHYfWOemZKLMb3chLWBC4b1z_B13hUPZtQvNPBR2ZufrY5078DV8i3PO1nl5YgEXXfmzmyMvHKpc8bob3imvvRPZydvJqDHeElGVMJojI58EJKfhiEZOuYQjFTmBFZYWPxHLhSTgn9pg9G_h2__gAyHgcM3feqdBFnZT5dvXCKl0NEzBClwrxTJifEb3GXirr4kLTGyoU6T7PU3cZiElPlLXCSYLLJWyve0UuikjaHdqo1ZcvhukzXvEr59IdaIAYLZV8Ow8UusAzIdsya44eEl3fIskX8yV7rCC3BkqRaA1Er7-0ZIp1d0hHNfAcC4cnXSzHFRoe.bI56SGu5-ejDQ6Gpbpns3Q/__results___files/__results___73_0.png)

Although this is a unique feature we can see that they have some connections from the above netwrok diagram.

### Distribution of Age over Race in dataset:
![alt_text](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FtNCcU2YJU0HMCcm2bYcfg.EeYTEdDyE67g9RhbahPsgEy5EZgamJQPVWdX5sSkjnUh4GSaX-pOXK4mOlrszCttad-fByawn5TMljC02DZwApd-svLUqTX4lO4PtyiT3gDelDcM5sV4JpdnTXnYxkv7xGpHOO8zCXLThMQoUGuXLmRA1tvORLqtcyxOGiTRYqwpEezq_8oIlJdmYy47HYC6bfaDNPJmKznegnB9q9EflyZyzv-xBHY--qNuXGRxb_K0L08-N-DCKKBnTKL2GmcSmbm_8coC-f986xzxhyO58r1QAF-YXAYSBM8GvcLd_zn8xBj91GLsSiEunstDjcn2pl3q3ziQdILBml6IzWISyXMo96KZz569MJiKPwWFXBhURfuiYYuHxbTUZxdveR9mhvGGdtu0x2aYpt4woJUsShzk0vqR-_-upv8vSzAEeApFObEYkfkGxsQJHn4PamalDwo4ZJV587GY7lXLAyKBD3YVtnNtzpfebVmhez5lhxzbDiarWhZnDCeKKiswNL1SWxQyjTceaCNb-wBdDbhFOS7dUKWAq8YoJs62f8jAJgOf17KICk8AvTTvEkTgTh_pVn1lQCHh6j-wQx9tJvs-l_kKRCJEsavjbBOhlMQv94rxuZxp6JP_KnU0Fdv25vJFwlQhmVuL7o6QFNBsfBdU4FSWzXCMiBzqK1acqxzMdv5rIz7VfeNFXdF7vrBhaUv2.ikIhLWKKC_d2KEJBK27THQ/__results___files/__results___100_1.png)

We can see that most of the black african or american people are around 50-60 ages from the above kde plot.We can see the same results in the violin plot which is shown below.

![alt_text](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..INqERxOS0gjQZ5ZR49elkA.1kNRtGM0HfoGJ4eW3xyFAu23xoQh80jI5_oOETHGu-Sjua0tyoRhCQUo9rAJLYDqEwdu1qjpiZBSrYUciQV5YBRiiArfbZts0FU-ya6feRpUgLCReeP1VOwG2pYnuj_d3CBh9F-yuTGo8Cxz6_pEuK6K9W6uxt35ck1AOC2MqUbEVADxFfSlumjTeLwQ0dAUemUlJlQMS4wRQf3YA0uUgMJTCJkMvsrxT12Klqj9IFhz4ow9vB33_gjDijwKAf1oHnya3Vn_Q5uMC49Hv9BGCl9BcoxMewBH2iAWq-fFY2DaLqoNV3-hmbDj6_L-KMsDFr9QU__Z8J4QRM-FahuGxIWoY-bSAGuVil2E8SFS32YRJo6XCONd3jFylb6Nz9JLjxXBGXLPC9X0jsB2L8TDGo3qcBuL6-aKoQjNBCw6tKAvsNSclAxJvJW0gJu3R5V-ojY3CBgvruD9rA_alaLQekHvrApSaNVVgt44J3aEY4wed3RSV2tEj1ucLttCFechovqTcGWPG5EEushCx95D5frEGQgSsHJGWIflp1T8xcjUgnkee2kF9ShSDHOalZ2DeEELN9A67NNLsQu-wa3RWmAkfl8jxFd5CEqefVBGpQKDnyetSCnR4qKJHbRCBZop2JxPJ9ufdUEOK8XORaabIxAfOYz_XItNw6S_YtvrSI35HDsLS0a8d5kTl8dF_5jZ.MXbyBysC70HdYIcsUJPb5g/__results___files/__results___101_0.png)

### HeatMap:
Finally the heatmap which is coded using seaborn below shows the correlation between each features in the dataset.

![alt](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0ILrOlSNQUVwxHzfdNk5aw.WE4JcR6vr9NfQoZ3FoVqvTZHpZ-gtt7OHvBPjSTHgd7Z3aunVxCmRYuiMUhEgoANEU_wlhjF6Ho4sMprZ-ekpdplJb9n3VySXVrPwQ-48BCyqibUcPpHmRIwdxcjbfVp7aS4FMUJnDOGXlzoCL2ATAJNhsovyehsilZKb4qF1EtNOvK2MztjX_FGF0cZKzQrQKX31y5vGAIsCxJhuWC88xTsvO3QmWFvZfCSmQPaNgkaZRXktvLWrWbEzF22Gty6iHvD-CWKy57vr8pnd-9yqfma4BEo7BXkhKiSRTaho_CmgV89hoE8Sq8Ln2DDCAuSWAG-mwoh9DplxIHEiE7M4648FvaO7g4WYflRAPEtkppt8ObCiBlifK_zrnZPzqC9KgwtnrG9KG6r4r-laUsJNF0fL1f5wim9kVkTCX15uwNEMJ9xXE1pIRkD77wEWokcVBEKWC6KGT7DJNaPSy5s5ifJpQDNlMwItBQc0Ki6QIvVJJEi5dW1MM0ArQ7fwnuYtptF9e4jgFt912i8EyhRtZVE5Y97K_8HeN5EQHXL0XQanwv2HqbOgZKGUJV1rW7V3jMfiHEbI3EyREYf9HjmbYCMNYkeRljjakYyEEAuxLChSX-j45t2jz4ATdXgmjkxBLJ9K7qZ6n4t8_xP7rqNpJzt9Cjtw1mjPL4j15kWr7XH9myYeF_wUr4kyTdNCt97.gYLR71XbGZAHl-SsLG8plA/__results___files/__results___121_1.png)

The code for producing this kinda correlation heatmap is given below:

      corrmat = hubmap_df.corr() 
      f, ax = plt.subplots(figsize =(9, 8)) 
      sns.heatmap(corrmat, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5)
 
 From the heatmap we can infer that weight and percent cortex are well-correlated.
 
 ### Visualizing Images:
 So the images were given in tiff file as said already which I converted to png format after removing masks from the image and they were visualized using matplotlib.
 
 ![alt](https://www.kaggleusercontent.com/kf/47295929/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..zZtA4RwvQG_s5zhmxYkP8w.9XsmmVIMNf38R9bqVVHA7CHbE814zeC5jWaqYmBxOfwTw_JPAaheT4vz-7fTg9ZfcHR6uARauvMZHGJFnP3htsibhOHrqTqYxVoOI6wePtWxdDJucSe4vFGtOeNOXUUezX9XwXO2c64lWcXHEULWR9aT3W1cDoXTsNnw56vZ8kTZxZJ2VT66szy0Tu1Wkms7JiEvOPhe143bcGjgu6S1wGVcrYJ_x-zTArK1llHFVbSwH0vTtzevNlGcFyCdcrs5p9T7jW2Ef4hddH5MHHdb0FSl10IiQ3BADZL1nps5QYtYPFFPsEmHy5qNq75tqQqGeQzoW6ySCzySNTsxCg7GHmlEbRcLvH4S6IpK45WZNAfMSgWSXUHNuxFBQijKYZxThreGzzMdxQLLTiI5H1i7RiqZZ7lneljlSOnLLtQMpcX0Ms10SdRcqMx-BHiLt02P_KQGMRfHeHHIQdjJBb-BmL0vetSmA_-ygxojQG7SH46b-OLWwhXcBVlsRDIxazhFu8PPdZE3NqFGxFIsuJ5Hk7dvFwA2WMZe4T_IJCa4n-f1AaUQ25DF9uxTdR_wx-WojQUStqAy_B2ps5T_Q2bt7p5XB8YtsZRZLrI_dmErhtIdQk3hQ9a7EAdBJggFWPbcpzi_Tia6u6qltC76c2X0GAV-hYJiScWjXNF6Wt9cBG5LV3jG41XKVvpocixbTB6X.Hm7zeJfZPtr2P8n3006iqQ/__results___files/__results___127_2.png)
 
 For the rest of the eda you can have a visit to my [kernel](https://www.kaggle.com/vineeth1999/hubmap-eda-pytorch-efficientunet-offline-training) and check for it.
