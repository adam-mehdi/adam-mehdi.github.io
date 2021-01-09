# Image Similarity Finder: Theory and Code

## Introduction

How can we compute similarity among image data? For tabular and vectorized data, we can compute similarity by simply finding the sum of the difference between each example 
along their categories. However, doing the same in images--summing the difference between each pixel value--would clearly fail. The information in images lie in the interaction 
between pixels; we would have to extract meaningful features out of the images into a vectorized form if we were to proceed in the same manner.
	
But how do we extract features out of unstructured data like images? In NLP, we use learnable embeddings--numerical feature vectors containing the meaning of particular words 
in a way that the computer understands. We can use vectors to represent words because words contain limited information in themselves. However, the adage a picture is worth a 
thousand words renders its taunting face here. Images are incomparably rich in information.

Even if we have a process whereby we can extract the features from images to compare them, how big would that feature vector have to be? Unfeasibly big. If 
we seek a method of comparing image similarity, then, the approach of converting images into a more structured form is likely doomed to fail. We need another approach, 
one that can parse and compare images in their entirety. Hence this project. 

I use a compound deep learning pipeline to propose an explainable heuristic for automatically finding similarity between images. I use the simple Oxford PETS dataset. 
The implementation of this pipeline is probably similar to that of facial recognition technologies, although I am unfamiliar with any other approach. In this article, I 
walk through each step of my project, from classification of pet breeds to finding similarity with the Siamese model and interpreting predictions with class activation maps 
(CAMs). The code is written using PyTorch and fastai. I will conclude by discussing potential applications of this heuristic as a crude clustering algorithm for minimally 
labelled datasets and matching similar patients for medical prognosis.

Here is the original project’s notebook. I suggest working through the notebook as you read through the following commentary, since I omit some details for brevity.

## Diving into the Implementation

Let’s begin where we can get a clear view of the whole project: at the end. 

The `SimilarityFinder` class is my modularized version of the inference pipeline, and once we understand its three methods, we will have grokked the essence of the project. `SimilarityFinder` strings together two models, a classifier that predicts the breed of a pet and a comparison (`Siamese`) model that determines whether two images are 'similar'. We use them to predict the image in our comparison image files that is most similar to the input image.

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
head = create_head(512*2, 1, ps=0.5)
smodel = SiameseModel(encoder, head)
```
4. Create the `Learner` and train the model. We deal with little wrinkles in `Learner`: specify the location of the body and head with `siamese_splitter` and cast the target as a float in `loss_func`. Note that after we customized the data and model, everything else falls into place, and we can proceed training in the standard way.
```
slearn = Learner(sdls, smodel, loss_func=loss_func, 
                splitter=siamese_splitter, metrics=accuracy)
slearn.fit_one_cycle(n_epochs, lr)
```

We use the capability of determining shared breed as a heuristic for image similarity. I use the probability that the two pets are of the same breed as a proxy for similarity: if the model is 95% confident that two pets are of the same breed, they are taken to be more similar than if the model predicts with 80% confidence.

### SimilarityFinder.predict

This is the most complex method in the project, so I'll break it down bit by bit. The jist is as follows: input an image file, predict its class, search through a repsitory of images of that same class, record activations of the body with a hook (for `similar_cams`), and output the most similar image.

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
          output = slearn.siampredict(ims)[0]

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
Now, let's return to the heart of the project, `SimilarityFinder`, in which we string these capabilities together.

