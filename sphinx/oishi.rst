Tracking in 2D: OISHI features
==============================


To locate the fish in a 2D image, we are looking at several points that "looks like the centre of a fish".

I call these centres oishi features, because they should be Orientation Invariant and SHape Invariant.

How to find these features? Shortly speaking, type these

.. code-block:: python

   import fish_track as ft

   kernels = ft.kernel.get_kernels(shapes, axis_indices, cluster_number)
   oishi_kernels = ft.oishi.get_oishi_kernels(kernels, 35)
   features = ft.oishi.get_oishi_features(image, kernels)


But what is happening? What are the meanings of those variables? What is the business happening inside the code?

Here is an detailed explaination.


Find a Shape in a Image
+++++++++++++++++++++++

under construction ...

Finding reprsentative Shapes
++++++++++++++++++++++++++++

under construction ...

Rotate the Kernels
++++++++++++++++++

under construction ...

Getting the OISHI features
++++++++++++++++++++++++++

under construction ...

Refining the OISHI features
+++++++++++++++++++++++++++

under construction ...
