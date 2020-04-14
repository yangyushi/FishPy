Tracking in nD: Linking
=======================


Using whatever trick you have in your notebook, finally you get some coordinates. Then you want the *trajectories* of the agents/fish/particles. How would you convert a collection of positions into trajectories?


Very briefly, I would write a script like below:

.. code-block:: python

   import fish_track as ft

   linker = ft.ActiveLinker(linker_range)  # alternatively, use ft.TrackpyLinker
   trajectories = linker.link(positions)
   trajectories_longer = ft.relink(trajectories, dx, dt, blur)

In the end, we get ``trajectories_longer`` which are long trajectories. The above code works very nicely in any dimension. (If you have experimental data for particles in 4 or higher dimension, please send me an email so I can worsihp your techniques.)

But what is happening? What are the meanings of those variables? What is the business happening inside the code?


The Paradox of Linking
++++++++++++++++++++++

under construction ...

The Colloidal Linker
++++++++++++++++++++

under construction ...

The Active Linker
+++++++++++++++++

under construction ...

Making Trajectories Longer
++++++++++++++++++++++++++

under construction ...
