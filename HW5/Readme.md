## Control Flow using opencv

In the above project two of the main algorithm for control flow have been utilized to detect the moving objects on the provided video.
There are two main categories of algorithms for motion detection on computer vision, Sparse and dense control flow algorithms.

1. Sparse methods

These are the methods that are mainly used in real time scenarios, because instead of working with all of the points in consecutive frames, it only requires some of the detected key points. It is notable that these methods can not be applied on all of the pixels in an image, especially in scenarios where the camera is dynamic. But generally these methods have promising results for real time cases, compared to dense flow control which are slower than these methods.

The algorithm which have been used in this project for sparse control flow is Lukas-Kanade algorithm. Using the derivate of the image with respect to time, horizontal and vertical axis it calculates the motion in two frames.

2. Dense methods

In contrast with sparse methods, these algorithms use all of the pixels in the image to calculate the motion. In the above project the FarneBack method, a well-known approach in this category, is used for motion detection. In this approach any neighborhood around a point x is approximated with a quadratic polynomial equation in both of the frames and by a little help of math and matrixes the global displacement of d is calculated.
