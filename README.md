
## How to run

All of the files to run the experiments reside in `Experiment` folder. The name of the python files start with the *experiment dataset*, followed by the *type of experiment*. For instance, you can run the *Health* experiment, and get the *average accuracy* by running:

```
 python Experiment/health_average_accuracy.py 
```

In general, you can run the experiments using the following rule:

```
 python Experiment/<dataset>_<experiment-type>.py 
```
  
 where `<dataset>` is either `health`, `cifar`, or `imagenet` (Note that, `camera` is only for *experimental* purposes with a distribtued neural network that is both vertically and horizontally split.), and `<experiment-type>` is either `average_accuracy`, `hyperconnection_weight`, `failout_rate`, or `skiphyperconnection_sensitivity`. 

The datasets and the preprocessing methods are explained in the paper. The experiments are as follows:

- `average_accuracy`: (Section 3.3). Obtains average accuracy, in addition to accuracy for individual physical node failures.
- `hyperconnection_weight`: (Section 3.4.1) Obtains results for different choices of hyperconnection weights.
- `failout_rate`: (Section 3.4.2) Obtains results for different rates of failout.
``skiphyperconnection_sensitivity`: (Section 3.4.3) Obtains results regarding which skip hyperconnections are more critical. 

## Dependencies

The following python packages are required to run these experiments:

- Keras
- sklearn
- networkx
- pandas
- cv2

## Output

Once you run an experiments, you will see the output in the console (e.g. accuracy). When the experiment finished running, new folders will be created. These folders keep the results and the models associated with each experiment. 

`/results` keeps all of the result text and log files from training.

`/models` keeps all of the saved models after training.


