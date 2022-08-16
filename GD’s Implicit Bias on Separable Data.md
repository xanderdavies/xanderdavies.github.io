# GD’s Implicit Bias on Separable Data

*TLDR: This post goes through the intuition behind why using gradient descent to fit a linear model to linearly separable data will learn a maximum-margin decision boundary. It’s a clean example of an implicit bias of gradient descent, and also extends (in some form) to more complicated settings (like deep networks).* 

Often, we use neural networks to solve **optimization problems where there are many different solutions which minimize the training objective. In these cases, the particular minima we learn (or approach) is a consequence of how we go about finding a global minimum; this is known as an *implicit bias* of our optimization algorithm. Since different global minima behave very differently outside of the training set, these implicit biases can have major effects on how our models generalize. Lots about implicit biases of different optimization algorithms still aren’t well understood; indeed, understanding these biases, and their corollary of consistently finding surprisingly well-generalizing solutions is a fundamental open problem in deep learning theory.

This post will talk about a particular setting where we *do* understand how this implicit bias is working. The setting is interesting not just as a toy case to develop intuition, but also because it seems like very similar versions of these results may hold in much more complicated settings. It’s also useful because it forces us to confront that inductive biases are playing a role in learning; thinking only in terms of loss-minimization doesn’t explain the observed behavior.

Lots of the time I find discussion of implicit bias to be unsatisfying or confused; understanding this example well will, I hope, clear things up. Most of this math is based on “[The Implicit Bias of Gradient Descent on Separable Data](https://arxiv.org/abs/1710.10345)” by Soudry et al.

## The Setting

Consider a classification data set $\{\vec{x}_n, y_n\}^N_{n=1}$, with d-dimensional real number inputs $\vec{x}_n \in \mathbb{R}^d$ and binary labels $y_n \in \{-1, 1\}.$ We use gradient descent to find a vector $\vec{w}$ which minimizes $\mathcal{L}(\vec{w})$, defined by:

$$
\begin{align*}

\mathcal{L}(\vec{w}) &= \sum_{n=1}^N \ell(y_n (\vec{w} \cdot\vec{x_n})) \\ \ell(u) &= e^{-u}

\end{align*}
$$

Our loss encourages high values of $\vec{w}\cdot\vec{x}_n$ which match the sign of $y_n$. We classify points according to the sign of $\vec{w}\cdot \vec{x}_n$. We assume our data is **separable** (that is, $\exists \vec{w}_*$ such that $\forall \vec{x}_n,$  $y_n(\vec{w}_* \cdot \vec{x}_n) > 0$). 

![Exponential loss, $\ell(u) = e^{-u}$.](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/Screen_Shot_2022-08-07_at_5.15.53_PM.png)

Exponential loss, $\ell(u) = e^{-u}$.

With these conditions, we know that the infimum of the loss is zero, but that this can’t be achieved by any finite $\vec{w}$. We use gradient descent to minimize $\mathcal{L}(\vec{w})$, with updates of the form:

$$
\vec{w}_{t+1} = \vec{w} - \eta \cdot \nabla \mathcal{L}(\vec{w}_t)
$$

With a sufficiently small $\eta$, we can prove that gradient descent will converge to a global minimum as $t \to\infty$. We make no assumptions about the initialization of our weight vector $\vec{w}$.

### The Task

In this setting, approaching zero loss requires the norm of $\vec{w}$ to diverge to $\infty$ (i.e., $||\vec{w}|| \to \infty$). However, since the sign of $\vec{w}^T\vec{x}_n$ determines its classification, blowing up the norm of $\vec{w}$ has no effect on the functional behavior of our model (and thus can’t effect things like the generalization behavior we care about).

On the other hand, the direction of our weight vector *does* determine classification, and thus is relevant to generalization behavior. Accordingly, we want to characterize the behavior of $\frac{\vec{w}_t}{||\vec{w}_t||}$ as $t \to \infty$. This distinction is important, and has interesting consequences like the potential for test loss to *increase* even as our classifier gets more accurate (we’ll talk about this in ‘Extensions’).

It’s worth focusing on the fact that a $\vec{w}$ which separates the data *doesn’t need to change it’s direction to approach a global minimum.* Scaling its magnitude is enough; accordingly, we could imagine it being the case that $\frac{\vec{w}
}{||\vec{w}||}$ doesn’t converge to anything in particular (and just sticks with the first separating solution it comes to while scaling up its norm). This isn’t what happens; instead, $\frac{\vec{w}}{||w||}$ does indeed converge. I focus on this to reinforce that this result *isn’t obvious,* and *isn’t what we come to by only reasoning about somehow getting to a global minima.*

### 2D Intuitive Model

There’s a really natural way to think about our setup with two dimensional points. Say we have a set of 2d points $\{\vec{x}_n = (a_n, b_n)\}_{n=1}^N$ with corresponding labels $y_n \in \{-1, 1\}$.

Now say $\vec{w}_0 = \begin{bmatrix} 0 & 1 \end{bmatrix}$. In this case, our decision boundary is the x-axis, since $\vec{w} \cdot \vec{x_n} = 0 \implies b_n = 0$. Here, $y_n(\vec{w}_0 \cdot \vec{x_n})$ measures the distance between a point and the x-axis, made negative for points on the wrong side of the decision boundary. This value is then scaled according to our exponential loss, and summed to get our $\mathcal{L}(\vec{w}_0)$.

What’s cool is that we can use this same intuition for all weight vectors $\vec{w}$, since we can always decompose $\vec{w}$ into a rotation ($\mathcal{R}$) and scaling ($\mathcal{s}$) applied to $\vec{w}_0$. And since $\vec{w} \cdot \vec{x}_n = (\mathcal{sR}\vec{w}_0) \cdot \vec{x}_n = \vec{w}_0 \cdot (\mathcal{sR}^T\vec{x}_n)$, we can equivalently think about new values $\vec{w}$ as performing a rotation and scaling to our data points, and then use our $\vec{w}_0 = \begin{bmatrix} 0 & 1 \end{bmatrix}$ value to score them as a function of their distance from the x-axis (visualized below).

![IMG_7818.png](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/IMG_7818.png)

![IMG_8168.png](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/IMG_8168.png)

![IMG_7604.png](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/IMG_7604.png)

![IMG_9366.png](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/IMG_9366.png)

Values of $\vec{w}$ which separate the data correspond to rotations which bring all points with a positive label above the x-axis, and all points with a negative label below the x axis. Scaling doesn’t change the classification, but it does change what’s plugged into $\ell(u)$, and thus our loss $\mathcal{L}(\vec{w})$.

Here, any separating value of $\vec{w}$ would give you zero loss if its norm was scaled to infinity; that is, *every separating rotation corresponds to an (unreachable) global minimum* achieved by diverging our scale $\mathcal{s}$ to infinity. The question of implicit bias, here, is the question of which separating rotation we get among the uncountably infinite number of separating rotations (which all can tend towards global minima via scaling).

## The Solution

So, what happens when we use gradient descent? How do we select between the uncountably infinite number of separating solutions? I think it’s good to think about this for a bit; not just guessing at what the solution will be, but *why* exactly gradient descent does this.

---

The key insight is that since $-\nabla \mathcal{L}(\vec{w})=\sum_{n=1}^N\exp(-\vec{w}\cdot\vec{x}_n)\vec{x}_n$, as the magnitude of $\vec{w}$ diverges to infinity (as $t \to \infty$), only the terms with the largest (least negative) exponents will meaningfully contribute to the gradient. These are the terms with the smallest margin ($\vec{w} \cdot \vec{x}_n$), which are the points closest to the decision boundary. The gradient, then, will be dominated by the closest samples scaled according to $\vec{w} \cdot \vec{x}_n$. This means that (as the magnitude of $\vec{w}$ diverges to infinity), our gradient $-\nabla \mathcal{L}(\vec{w})$ will become a non-negative linear combination of the closest points (support vectors). 

At this point, I think it becomes intuitive that we would converge to the decision boundary preferred by these closest points, since these closest points have total control over our updates. The decision boundary preferred by these closest points, $\argmax_{\hat{w}}(\min \hat{w}
 \cdot \vec{x}
)$, is the max-margin solution.

In our 2D framing, this means selecting the rotation (without scaling) which *maximizes the minimum height* of the points (and still correctly classifies). This is different from other plausible answers, like minimizing the sum of distances from the x-axis across all points.

![IMG_3799.png](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/IMG_3799.png)

One thing that’s cool is that this [turns out](https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/max-margin.pdf) to be the same as the direction of the $\hat{w}$ with the smallest $\ell_2$ norm which maps all points to at least distance one from the decision boundary (on the correct side):

$$
\begin{equation} \vec{w}^* = \argmin_{\vec{w}}
||\vec{w}||^2
 \text{ s.t. } y_n(\vec{w}\cdot \vec{x}_n) \geq 1 \end{equation}
$$

Again, this is intuitive in our 2D model; as we rotate and then attempt to scale down as much as possible, we are stopped by the points closest to the x-axis. The rotation that allows us the most scaling is the rotation which initiates these closest points furthest from the x-axis, which is the max margin solution. 

![Our down-scaling is blocked by the closest points intersecting with $y=\pm 1$.](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/IMG_4082.png)

Our down-scaling is blocked by the closest points intersecting with $y=\pm 1$.

It’s important to note that we only get to this solution because of the exponential tail of our loss function, since this is what made all but the support vectors’ gradients not matter.

---

To be (a bit) more formal, though, we can say that if $\frac{\vec{w}}{||\vec{w}||}$ converges to some value $\vec{w}_\infty$, then this $\vec{w}_{\infty}$ must itself be dominated by a non-negative linear combination of its support vectors; the part of $\vec{w}_\infty$ which does not come from these support vector gradient updates (i.e., the initial conditions) is negligible (since its norm tends to infinity). This $\vec{w}_{\infty}$ is proportional to $\hat{w} = \frac{\vec{w}_{\infty}
}{\text{min}_n(\vec{w} \cdot \vec{x}_n)}$, which has the properties:

$$
\hat{w}=\sum_{n=1}^{N} \alpha_{n} \vec{x}_{n} \quad \forall n\left(\alpha_{n} \geq 0 \text { and } \hat{w} \cdot \vec{x}_{n}=1\right) \text{ OR } \left(\alpha_{n}=0 \text { and } \hat{w} \cdot \vec{x}_{n}>1\right)
$$

That is, $\hat{w}$ is a non-negative combination of points distance one from the decision boundary, and all other points are further from the decision boundary. These turn out to be the [KKT](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) conditions for eq. 1, meaning $\hat{w}$ is its solution. Since $\vec{w}_{\infty}$ is proportional to $\hat{w}$, we have that $\frac{\vec{w}}{||\vec{w}||}$ does indeed converge in direction to the max-margin solution.

## Extensions

This result has been extended to a broader class of loss functions (including multi-class classification with cross-entropy loss). It’s also been extended to using *[stochastic* gradient descent](https://arxiv.org/pdf/1806.01796.pdf), and to [positive homogeneous deep networks](https://arxiv.org/abs/1906.05890) like ReLU MLPs. Soudry et al. expected this to be true; the below figure is taken from their paper, and shows a convolutional neural network trained on CIFAR10 using SGD.

![Screen Shot 2022-08-07 at 6.40.18 PM.png](GD%E2%80%99s%20Implicit%20Bias%20on%20Separable%20Data%20d37d962d88f943689a5fefaa46ce6154/Screen_Shot_2022-08-07_at_6.40.18_PM.png)

This figure initially looks weird, since the validation loss goes *up* as we train past separation, even as the validation accuracy improves. In light of these results, though, this is intuitive; though we are converging towards the max margin direction (which should *help* accuracy), we also blow up our weight norm (which amplifies any mistakes we make). Sometimes, though, we mistakenly think this increase in validation loss suggests our model is generalizing worse and worse, and decide to stop training accordingly (whereas using validation accuracy would correctly lead us to keep training).

What about adding a bias term to our original setup, learning according to $\ell(y_n(\vec{w} \cdot \vec{x}_n + b))$? One way to do this is to appending a one to each input, forming $\vec{x}_n'
 = \begin{bmatrix} x_{n, 1} & ... &x_{n, d} & 1\end{bmatrix}$. Of course, our same results hold for $\vec{x}_n'$, meaning we get the max margin solution (but in the new input space). In our 2D case, we can think of this as converting all of our points into 3D points with $z=1$, and then allowing 3D rotations instead of just 2D ones. We then get the max margin solution in this 3D space.

### Other Optimizers

Momentum, acceleration, and stochasticity all don’t change this implicit bias. Interestingly, though, it turns out that adaptive optimizers like Adam and AdaGrad don’t converge to the $\ell_2$ max margin predictor, and instead the limit direction can depend on the initial point and step size (though they do converge to zero loss). This might be part of the reason that these optimizers are thought to find solutions which don’t generalize as well as SGD.

## Conclusion

If we understood what sorts of solutions gradient descent converged to, and what generalization properties those solutions had, we could feel more confident in predicting traits about our learned model. In particular, if we *knew* our model would converge to the max-margin solution of the training data, and we *knew* that max-margin solutions tend to generalize in particularly nice ways, we would expect our model to also generalize in nice ways (in the limit). I think this has ramifications relevant to AI safety (more info in a subsequent post), and is part of the reason I think studying the science of deep learning is useful.