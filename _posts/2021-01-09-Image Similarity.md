# Image Similarity: Theory and Code

A heuristic for computing similarity in unstructured image data.

## Introduction

How could we compute how similar one image is to another? For similarity among data in a vectorized form, we can find the sum of the squared differences between two examples, or use similar methods like cosine similarity. However, performing such techniques on images--summing the squared difference between each pixel value--fails, since the information in images lie in the interaction 
between pixels. We would have to first extract the meaningful features out of the images into a vectorized form if we were to proceed.
	
But how do we extract features out of unstructured data like images? In NLP, we use learnable embeddings--feature vectors containing the meaning of particular words. We can use vectors to represent words since words contain bounded information in themselves. However, that won't work in images. As the hackneyed adage goes: *a picture is worth a thousand words*. Images are incomparably rich in information, so representing them in vectors is no trivial pursuit. Even if we had a process that extracts vectorized features from images, that feature vector have to be extraordinarily big. Unfeasibly big. 

If we seek a method of comparing image similarity, then, the approach of converting unstructured images into a structured form is likely doomed to fail. We need another approach, one that parses and compares image information directly from the unstructured images. Hence this project. 

I propose a compound deep learning pipeline as an explainable heuristic for automatically predicting similarity between images. To do so, I used the Oxford PETS dataset. 
The implementation of this pipeline is probably similar to that of facial recognition technologies, although I am unfamiliar with those approaches. In this article, I 
walk through each step of my project, from classification of pet breeds to finding similarity with the Siamese model and interpreting predictions with class activation maps 
(CAMs). The code is written using PyTorch and fastai. I will conclude by discussing potential applications of this heuristic as a crude clustering algorithm for minimally 
labelled datasets and matching similar patients for medical prognosis.

Here is the [original project’s notebook](https://colab.research.google.com/drive/1LQ6gbZHioY09GQRBh99VgD-Lxc8L5lv4?usp=sharing). I suggest working through the notebook as you read through the following commentary, since I omit some details for brevity.


<img src="/images/title_cam.jpg" width="500" height="224" title="A pair of pets predicted to be similar">


## Diving into the Implementation

Let’s begin where we can get a clear view of the whole project: at the end. 

The `SimilarityFinder` class is my modularized version of the inference pipeline, and once we understand its three methods, we will have grokked the essence of the project. `SimilarityFinder` strings together two models, a classifier that predicts the breed of a pet and a comparison (`Siamese`) model that determines whether two images are similar. We use them to predict the image in our comparison image files that is most similar to the input image.

``` python
class SimilarityFinder:
    def __init__(self, classifier_learner, siamese_learner, files):
    def predict(self, fn, compare_n=15):
    def similar_cams(self):
```

In `__init__` we preprocess the image files that we are using for comparison into `lbl2files`, a useful mapping for `predict`, and initialize our two `Learner`s. A `Learner` is a fastai class that wraps the model, data, and a few other training components into a single class, so we can think of them as the two parts of our pipeline. 

```python
def label_func(fname):
    """extracts the pet breed from a file name"""
    return re.match(r'^(.+)_\d+.jpg$', fname.name).groups()[0]

class SimilarityFinder:
    def __init__(self, classifier_learner, siamese_learner, files):
      self.clearn,self.slearn = classifier_learner,siamese_learner
      labels = L(map(label_func, files)).unique()
      self.lbl2files = {l:[f for f in files if label_func(f)==l] 
                        for l in labels}
```

The classifier `Learner` will serve as a heuristic for reducing the amount of images we have to sift through in predicting similarity. The Siamese `Learner` predicts similarity between two images. Together, they will allow us to find the most similar image in a sizeable dataset.

Let's continue by looking at how we built those two `Learner`s.

### Classification

We predict the pet breed from images of pets. This is a standard classification problem, so it should seem trivial to those familiar with CNNs. There are three basic steps:
1. Extract the image files from a directory. The PETS dataset is available by default in the fastai library, so we use `untar_data` to access it.
```
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
```

2. Preprocess the image files and store them in `DataLoaders` with fastai's Data Block API.
```
cdls = DataBlock(blocks = (ImageBlock, CategoryBlock),
                get_items = get_image_files,
                get_y = using_attr(RegexLabeller(r'(.+)_\d+.jpg$'),'name',
                splitter = RandomSplitter(),
                item_tfms = Resize(224),
                batch_tfms = aug_transforms()).dataloaders(path/'images')
```
3. Wrap everything in a fastai `Learner` and train the model. I used a couple tricks for training (label smoothing, mixed-precision training) in the project, but I omit them here for simplicity. They are available in the original notebook.
```
clearn = cnn_learner(cdls, resnet34, metrics=accuracy)
clearn.fit_one_cycle(n_epochs, lr)
```

The classification pipeline is complete; let's move on to the more complicated comparison pipeline.

<img src="/images/classifier_show_results.jpg" width="500" height="224" title="Showing the results of the classifier">

### Comparison

We trained a model to predict pet breed. Now, we train use a model that predicts whether two images are of the same breed. It will require defining some custom data types and a custom model, as it is not a standard application. The following implementation is drawn from the Siamese tutorial on the fastai documentation, but I made modicications on the model and training process.

Implementing the Siamese model is very similar to implementing the classifier; however, there are two key modifications. 

We input two images into the model instead of one. This means that, firstly, we need to represent our `DataLoaders` with three elements per example--first image, second image, and whether they are similar--and, secondly, we pass each image individually through the same body and concatenate the outputs of the body in the head. 

1. Exactly as before, retrieve the image files.
```python
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
```
2. Preprocess the data with fastai's mid-level API. We create a `Transform` that opens files, pairs them with others, and outputs it as a `SiameseImage`, which is essentially a container used to display the data. Then, we apply the necessary transforms on all files with `TfmdLists` and `dataloaders`. 
```python
class SiameseTransform(Transform):
      def __init__(self, files, splits):
      """setup files into train and valid sets"""
      def encodes(self, f):
      """applies transforms on f and pairs it with another image"""
        f2,same = self.valid.get(f, self._draw(f))
        im1,im2 = PILImage.create(f),PILImage.create(f2)
        return SiameseImage(im1,im2,int(same))
      def _draw(self, f, splits=0):
      """retrieve a file--same class as f with probability 0.5"""
splits = RandomSplitter(seed=23)(files)
tfm = SiameseTransform(files, splits)
tls = TfmdLists(files, tfm, splits=splits)
sdls = tls.dataloaders(after_item=[Resize(224), ToTensor], 
        after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
```

3. Build the Model. We pass each image in the pair through the body (aka encoder), concatenate the outputs, and pass them through the head to get the prediction. Note that there is only one encoder for both images, not two encoders for each image. Then, we download some pretrained weights and assemble them together into a model.
```python
class SiameseModel(Module):
      def __init__(self, encoder, head):
          self.encoder,self.head = encoder,head
      def forward(self, x1, x2):
          ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
          return self.head(ftrs)
encoder = create_body(resnet34, cut=-2)
head = create_head(512*2, 2, ps=0.5)
smodel = SiameseModel(encoder, head)
```
4. Create the `Learner` and train the model. We deal with little wrinkles in `Learner`: specify the location of the body and head with `siamese_splitter` and cast the target as a float in `loss_func`. Note that after we customized the data and model, everything else falls into place, and we can proceed training in the standard way.
```
slearn = Learner(sdls, smodel, loss_func=loss_func, 
                splitter=siamese_splitter, metrics=accuracy)
slearn.fit_one_cycle(n_epochs, lr)
```

We use the capability of determining shared breed as a heuristic for image similarity. I use the probability that the two pets are of the same breed as a proxy for similarity: if the model is 95% confident that two pets are of the same breed, they are taken to be more similar than if the model predicts with 80% confidence.

Now, let's return to the heart of the project, `SimilarityFinder`, in which we string these capabilities together.

<img src="/images/siamese_show_results.jpg.JPG" width="370" height="224" title="Showing the results of the Siamese model">

### `SimilarityFinder.predict`

This is the most complex method in the project, so I'll break it down bit by bit. The gist is as follows: input an image file, predict its class, search through a repository of images of that same class, record activations of the body with a hook (for `similar_cams`), and output the most similar image.

```python
class SimilarityFinder:
  def predict(self, fn, compare_n=15):
    self.preds,self.acts,self.images,self.fns = [],[],[],[]

    # 1. predict breed of input image
    cls = predict_class(fn,self.clearn)

    # 2. retrieve a list of same-class images for comparison
    compare_fns = self.lbl2files[cls][:compare_n]

    # 3. register a hook to record activations of the body
    hook_layer = self.slearn.model.encoder
    with Hook(hook_layer) as hook:
      for f2 in compare_fns:

          # 4. preprocess image files for comparison and predict similarity
          im1,im2 = PILImage.create(fn),PILImage.create(f2)
          ims = SiameseImage(im1,im2)        
          output = slearn.siampredict(ims)[0][1]

          # 5. record state and outputs
          self.preds.append(torch.sigmoid(output))
          self.fns.append((fn,f2))
          self.images.append((im1,im2))
          self.acts.append(hook.stored)
          hook.reset()

    # 6. retrieve most similar image and show it with original
    self.idx = np.array(self.preds).argmax()
    sim_ims = self.images[self.idx]
    title = f'{self.preds[self.idx].item()*100:.2f}% Similarity'
    SiameseImage(sim_ims[0], sim_ims[1], title).show()
    return self.fns[self.idx][1]
```
1. Predict breed of input image. `predict_class` does preprocessing on an image file and outputs the predicted class using the classifier model.
```python
def predict_class(fn,learn):
      im = first(learn.dls.test_dl([fn,]))[0].cpu()
      with torch.no_grad(): output = learn.model.eval().cpu()(im)
      return learn.dls.vocab[output.argmax()]
```
2. Retrieve a list of same-class images for comparison. I am using predicted class as a heuristic to reduce the amount of images we must search through to retrieve the most similar. `compare_n` specifies the amount of images we would search through, so if case we want speedy results, we would reduce `compare_n`. If `compare_n` is 20, calling `predict` takes about one second.

3. Register a hook to record activations of the body. Hooks are pieces of code that we inject into PyTorch models if we want them to perform additional functionality. They work well with context managers (`with` blocks) because we must remove the hook after using it. Here, I used the hook to store the final activations of the model's body so I could implement `similar_cams` (explained later).
```python
class Hook():
      def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)
        self.stored = []
      def hook_func(self,m,i,o): self.stored.append(o.detach().cpu())
      def reset(self): self.stored = []
      def __enter__(self,*args,**kwargs): return self
      def __exit__(self,*args,**kwargs):  self.hook.remove()
```
4. Preprocess image files for comparison and predict similarity. `SiameseImage` is a modified tuple used to group and show our images. The `siampredict` method is a version of `Learner.predict` with modified defaults to deal with some wrinkles with the custom model.

5. Record some statistics.

6. Retrieve the image pair with the greatest predicted probability of similarity, taking them to be the most similar of the images considered. Show the images side-by-side with `SiameseImage.show` and output the file name of the most similar image.

That is the primary functionality of the pipeline, but, if implemented as such, we would not know why the images were considered the "most similar". In other words, it would be useful if we could determine the image features that the model utilized to make the prediction. Lest the model predicts two images to be similar due to extraneous factors (i.e. similar backgrounds), I added a CAM functionality.

<img src="/images/simfinder_predict1.jpg" width="400" height="224" title="Outputs of predict">

### CAM

Class activation maps are grids that show the places on the original image that most contribute to the output. We create one by matrix multiplying the activations of the model's body (called a spatial map) with a matrix containing the gradient of the output. Here, I used the weight matrix of the final layer of the model as the gradients, as the derivative of the output with respect to the input of the final layer is the final layer's weights. 

Intuitively, the spatial map shows the prominence of the features in each position of the image, and the gradient matrix connects each feature with the output, showing the extent to which each feature was used. The result is an illustration of how each position in the image contributed to the output.

```python
class SimilarityFinder:
  def similar_cams(self):
    # 1. grab the final weights and spatial maps of the most similar images
    sweight = self.slearn.model.head[-1].weight.cpu()
    act1,act2 = self.acts[self.idx]
    # 2. matrix multiply the weights and spatial maps
    cam_map1 = torch.einsum('ik,kjl->ijl', sweight, act1[0])
    cam_map2 = torch.einsum('ik,kjl->ijl', sweight, act2[0])
    # 3. open the most similar images to show them
    f1,f2 = self.fns[self.idx]
    t1,t2 = to_tensor(f1,slearn.dls),to_tensor(f2,slearn.dls)
    # 4. show the CAMs overlain on the images
    _,axs = plt.subplots(ncols=2)
    show_cam(t1,cam_map1,axs[0])
    show_cam(t2,cam_map2,axs[1])
```
1. Grab the final weights of the Siamese model as well as the spatial maps of the most similar images, which we recorded with the hook in `predict`.
2. Perform the dot product between the weights and spatial maps with `torch.einsum` (a method of custom matrix multiplications).
3. Open the files predicted to be the most similar in `predict`, and convert them into preprocessed tensors that we will be able to show.
4. Overlay the CAMs on the original images and show them side-by-side.
```python
def show_cam(t, cam_map, ctx):
      show_image(t, ctx=ctx)
      ctx.imshow(cam_map[0].detach().cpu(), extent=[0,t.shape[2],t.shape[1],0], 
      alpha=.7, interpolation='BILINEAR', cmap='magma')      
```

<img src="/images/simfinder_cam.jpg" width="400" height="224" title="The output of show_cam">

## Final Words

In this project, we predicted the most similar pet and then interpreted that prediction with CAMs. To conclude, I will attempt to more precisely define "most similar" and explain why this nuanced definition holds practical consequences.

The central insight in this project is that we can use a Siamese model's confidence in a prediction as a proxy for image similarity. However, "image similarity" in this context does not mean similarity in images as a whole. Rather, it refers to how obviously two images share the features that distinguish a target class. When using the `SimilarityFinder`, then, the classes with which we label our images affect which image is predicted to be the most similar. 

For instance, if we differentiate pets with breed as we did here, the `SimilarityFinder` might predict that two dogs sharing, say, the pointed nose that is distinctive of their breed, are most similar even if their other traits differ considerably. By contrast, if we are to distinguish pets based on another class, such as whether they are cute or not, the model might consider similar floppy ears more in its prediction than a pointed nose, since floppy ears would contribute more to cuteness. Thus, `SimilarityFinder` overemphasizes the features that are most important to determining the class on which it is trained.

The variability in predicting image similarity based on training label is a useful feature of `SimilarityFinder` if we are to apply it to more practical problems. For instance, `SimilarityFinder` would be a useful heuristic for finding the similarity between CT scans of pneumonia patients, as that similarity measure would help evaluating treatment options. To illustrate, if we can find the past patient with the most similar case of pneumonia and they responded well to their treatment, say, Cleocin, it is plausible that Cleocin would be a good treatment option for the present patient.

We would determine the similarity of the cases from the CT scan images, but we do not want the model to predict similarity due to extraneous factors such as bone structure or scan quality; we want the similarity to be based on the progression and nature of the disease. Hence, it is useful to determine the features that will contribute to the prediction by specifying class label (e.g. severity and type of pneumonia) and to confirm that appropriate features were utilized by analyzing our CAMs.

The purpose of this project was to implement an algorithm that can compute similarity on unstructured image data. `SimilarityFinder` serves as an interpretable heuristic to fulfill that purpose. For now, I am interested in applying that heuristic to medical contexts, providing extra data for such clinical tasks as matching pairs for interpretation of randomized control trials. More to come in subsequent posts.

### References
1. [Deep Learning for Coders with Fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)
2. [fastai documentation](https://docs.fast.ai/)
3. [grad-CAM paper](https://arxiv.org/abs/1611.07450)
