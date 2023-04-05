## Paper review

Here we review [this article](https://ai.googleblog.com/2022/07/simplified-transfer-learning-for-chest.html) from the Google AI Blog.

It talks about a new approach of transfer learning in the domain of Chest X-Rays (a.k.a. CXR). 
It reaches interesting conclusions, applicable outside the CXR usecase.

### Article summary

In classical transfer learning, a generic model is trained on a very large image dataset to extract the generic features
of any image. In other words, it converts any image to embeddings (i.e. information-rich numerical vectors). This first step 
is done on a very generic dataset not specific to any domain (such as ImageNet).
Then in order to perform a specific task, a fully-connected neural network is appended at the end of the above pre-trained model
to be trained on a very specific supervised task using a labeled, specific, dataset.

The issue with this approach is that the pre-trained model being non domain-specific, it is not good at extracting features
from images of this very particular domain, so it still requires a large labeled dataset to give a good accuracy. 
For certain tasks, such as medical diagnosis, such datasets are not available, or very expensive to build.

The article proposes a new approach to tackle this issue : inserting a second pre-training between the generic pre-training and 
the specific supervised task. This intermediary training would be done on domain-specific images, that is CXRs, but that
don't need to be labeled for a specific task. Thus, this second set of layers produces an embedding specific to CXRs and 
can then be used for any supervised task in the CXR domain (e.g. identify patients with tuberculosis, airspace opacities, COVID 19 etc). 

The authors go into more details by suggesting to train the intermediary pretrained model using Supervised 
Contrastive Learning (SCL). In this method, the objective is to learn representations that not only separate 
different classes but also group similar examples within the same class. The model is trained to maximize 
the similarity between representations of the same class while minimizing the similarity 
between representations of different classes. 

The article shows very promising results : after training the domain-specific embeddings model on a dataset of 800k Chest X-Rays,
the authors were able to train a tuberculosis detection model that matched the same accuracy (the actual metric used is 
AUC to measure discrimination) than radiologists with only 45 labeled images !


### What I liked about this article

I liked many things about this article :
- The research has significant real-world implications, as it can potentially enhance the analysis of chest X-rays and hence improve medical diagnosis
- It combines two existing methodologies (transfer learning and SCL) to propose a completely new groundbreaking approach
- The resulting approach can actually be applied to any other fields involving domain-specific images with a variety of subsequent tasks, such as spot spraying ;)
