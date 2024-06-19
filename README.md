# Handwritten-Digit-Recognition
## _Infosys Springboard - Handwritten Digit Recognition with LeNet5 Model in PyTorch_


[![Infosys Springboard](https://infyspringboard.onwingspan.com/web/assets/images/infosysheadstart/app_logos/landing-new.png)](https://infyspringboard.onwingspan.com/web/en/login)

An Internship project of Infosys in the feild of Artificial Intelligence, on Handwritten Digit Recognition
Using various Deep learnig Architecture and then deploying the model



## Features

> Trying various deep learnig models 
> Improving model accuracy
> Deploying all models together in one web app
> Making evaluation Report of the project

## Tech

We uses a number of open source projects to work properly:

- [LeNet-5](https://www.kaggle.com/code/blurredmachine/lenet-architecture-a-complete-guide) - A  Simple CNN Architecture
- [PyTorch](https://pytorch.org/) - A Machine Learning Library
- [MNIST](https://paperswithcode.com/dataset/mnist#:~:text=The%20MNIST%20database%20(Modified%20National,test%20set%20of%2010%2C000%20examples. ) - Modified National Institute of Standards and Technology database Containing Handwritten Digits
- [Extended MNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) - EMNIST Dataset Is A Set of Handwritten Character Digit
- [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network#Architecture) - Convolutional Neural Network Architecture
- [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) - Multilayer Perceptron
- [Streamlit](https://streamlit.io/) - Transforms ML Models Into Engaging Web Apps



## Installation

import Libraries as according_to_project :
- [![PyTorch](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAJFBMVEVHcEz0Ti3tSyvsSyvyTSzuTCvrSyvvTCztSyvvTCzyTSz2Ti1jX8EMAAAAC3RSTlMA61QhuW4Qgjed1ICuvi8AAADFSURBVCiRfZJZEsQgCETFBVnuf99pk5oMmHL4SAofS4OWkozL2WY9s+FHyIPOEMyllNYObIB1e6dXv9j6z7eWmxWeutdFPI3gSmBCTpNDaP9lc8+NNLqVvMfFMbl9C7NeA2Z5XwXN3LK+Rq78VNVNfX8kAW5zo9EZhkwx71umPRCCKEMJg2KUfBPq9BygqcUlNI/70+StTYdKEOBW72RGnawempx0iNSBS9h3Iop4J1pfez2FVvsCQCo7W81Qc47a/r36YB9hggYfnisLkgAAAABJRU5ErkJggg==)](https://pytorch.org/) 
- [![Numpy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAATlBMVEVHcExNlc9Nq89Nq89Nq89Nq89Nr89Nq89Nq89Nns9Nps9Nq89Nq89Nq89Nq89Nd89Nq89Nd89Nd89Ndc9Nd89Nd89Nd89Nd89Nd89Nd89FlH+4AAAAGnRSTlMAH11ouOwwrf4KFMZ8TZw81qrNXbn/UHjrjGxZf74AAAEJSURBVHgBrZJFggQxCEWBWPMz5X7/i05Iu+2a8no40G+ERfgbcj4E7z7jmE5FfHwD6pIAksMfa9M6fUCQXGyyKFi7FOwVN9j05jC0IFWp731zddkxXN+3UWPrG8Dl3gFRK+RhnIgNBUuoIWZtUstnOC/Las4tsEUmKmr+Dmdzzy5nx2gizMMdLh1hm7Q6DPIKB3vMDCukwoQ7XDqURxctrCDkRukK930Z+A5VwBFXOA7LPg1nGHwkdjnxDXbFtFoGL0zqHxMa6SiuC4Qwu5dSRuWCakEun+QB6rSPWkwNwrpUYXudC1YlHpeHUnxDj6LbHWqjryt0LSVFehfexlJKEtAn0Q5oQL+Qf6D1EW3rF4ehAAAAAElFTkSuQmCC)](https://pytorch.org/) 
- [![Pandas](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABDElEQVR4AWPADhqYpBh8uRgY0liRRUXYQ7YLs4f8YWBwYGHAB4Q5g82BCv8LcwS3I4sDxXaBxEEG08YAAYYAAXGGWG5kA1QYPNkFGUL5Qc7GawBIECQJUiTAEaEApKeKsIf6CbOFVoDEhdhD3Ig24P9JBuH/pxhigNiAPAPOMJgDNf8H4nayXQDEUUCsL8wR4gwUmwj0jjpJYSDEHjxTmC044AVbh+5z1vaE1wydUkQbgBwLL1g6Kp6zdPx/ytLuRpYBz9jaNJ4zd4S9ZGgVJ8sAIfZQD2ASni3EFqpFtAFADiMoATEwhDIjxwIokYESG0g1AQMQANkAmBhtDICm9T+gLIsmwQiSg9CYAADqzp3Adb0u+gAAAABJRU5ErkJggg==)](https://pandas.pydata.org/)
- [![Matplotlib](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAllBMVEVHcEzi6u7M19+/ztj////b4+n////////////t8fT////////////7+/v4+PnK1t//ol/l6eyswc7d3N3Iy8P/3GTx8vXQ3OTU1dT61VedtcWnvcu5y9TrkVDMuHnFsJzUh1DMkGnMwbSDenG4wprEmX6b3Jnk/3es/aqrzqrdvli70XDI42eHnqfX9mxZuqWewblGYp5W5il+AAAADHRSTlMAw+r+Yes0QhvdhocFXJSOAAABkUlEQVQokX2Th3bDIAxF7bjZYhmCSWInHtmj4/9/rpJwuk85eF4kpCeRJI8xmqeDLBuk81HyczwN17ky0huVr4dP39BkuirAWgABONRqOvlkY5VD2+g9CCkY52r8wcrCNFrrmiwF46Ls6UQVECqElSG3FiQuKVT0PM3RokWonfAQbjuPD8inHOcKmYAtwi1a7o7Pjl2vKOah4iDFXusGhL8dd/glMeYh5r6GmIRvcFPhkCGh5etRMscdN8tNADC1DnC9erbDmc+SVIG/LBaEQ9WaqxVxFwEqTQYG7HKxYOz2uxARTTNIMgmOIeGuu78J2fuVZZJ5gsvl5bJpu66zmKSgKYUv2e1m2wYvy7t1rjNf3aYxTbAvrxLOztTtl4AoFUkfATc7nMFVpFOfColAlpI0PxxIx8Y8RGD5KDq83OmA972uA0T5euE5+PPpFPAVddz2wnPJPDcARkseTaVrH0tGxcbqktr9EK62j2JTm9AvydnxQ360SWwwwuKPBuPWVNHj79b8v6n5OMzicZh9Hod3ftIqdrDY334AAAAASUVORK5CYII=)](https://pytorch.org/) 
- [![Streamlit](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAApVBMVEX////ox8j89/e5Ky//1tbIa2y7OTzi19f/S0vkvb29QEONVFj/vb3/CQn/AAD/YWH/4uLq4uOXZmpxFB5zHCXQvb7/Kyv/ISH/cnL/6+vx7Oyhdnl5LTN9NTv/fn7/Hh6qhoj/EhLnMjS7QEN8Ji3/Pz/jODq1QER8MjiEREn/rq7/FxfaOz2ePULFra//eXn8Li/RPkD2MjLwAAD/xMT/mpr/iopqCY8wAAAA00lEQVR4AdXO1Q3DUABDUYc5ZWaGMO4/Wh+V2wFyf49kGQ1Okv6brCgyXlPxTNN1Dc8MmOaDLcW2FetBjgPX81ttMWqTxHCn2+sPAH848scTNkqRDU9nvfmiB2C5GhJeymSUpVjymtBiviao+kPSarO1d1z1/XxB6hsgHYai447zidr5Ato1GN4LOUcE9zFDmew++USZYE8GKxm9aJoRPi3mM/BUT8CD7YjcEaXD9/Jsd8Y91/e9TbBajUirVbDx/MLCs4laXqt6mSTLurqW6gQN7AYfFRhollEAsQAAAABJRU5ErkJggg==)](https://streamlit.io/)


## Team

Countributers and Teammate in this project

| Members | ![Github](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAb1BMVEX////4+Pi3ubtvcnZNUVU+Q0cpLjLr6+x3en0sMTYkKS59gIORk5aUl5n8/Pzw8PFTV1tbX2Pc3d5DSEzn5+g3PECLjpFKTlKFh4qxs7XCxMUwNTq/wcLh4uPV1tZzd3o/Q0jOz9CmqKpjZ2qfoaSrd37mAAABPUlEQVR4AW3TBZKEMBAF0B8GCHzcnbW5/xm30qEyknklcU/DgQpuYRTHUXgLFHw6SemkmcYrlcd8kRYlnlQ1PU0Fp434Qde75Qd+1FUQKiRZjyGfTGNjKhWMmSQXYO3Ibao3MlqBnSRzADhk/ycAdcqclSSHnEUD+KLt8KalMQMqpl3izU5jKxHQGCq8Ud80fq4VfuFZaIyQO4wVPEre5g+RrIAPJrkQSL8OPjv3htQmH8guU5uwgseeP7ITMYBnpdFgvlJPcx0zoLjjzS/FDrVRvH6xsqDYlLx29huRUaFx6YuI1mhKMbddf9trEzca7rmRk/FxpiRXiJO8FDBURyb4yfO7glC8TOpacmAc4ElMEWlc2oGckjwvYVFEB5wjouE6uLBwquypQym/scKrM4njElYaJy182q15aDj/oQMZkS8JH3IAAAAASUVORK5CYII=) | ![LinkedIn](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABDUlEQVR4AWP4////gOLB44D6nTcsGIo33QHi/zTGd0B2YTiAPpYjHIHNAf/piQk6wGPW8f/rLz8HYRCbXg5AWI4GQGJ0cwDY12gAJDbcHUA4CkZAIqQUK7Ts/m/SfxBMs5RupswBaACr+P47b/5zlG/5DyzZ/r/+8hNF7vuvP//nn3r0X6JhJ+0ccPrR+/+H7735jw9cf/n5v0D1Nuo5gBxQve06zR0AjoL7b7/+//zjN4bc+ScfaOeA33///k9Yfg4mDw7u/Xdeo6uhnQP6D93FMNxlxjF0ZbRzgMXEQ9iyI90cALIMJoccDXRzAK6CZog6YNQBow6gIx54Bwx4x2RAu2bAysoEZu9o7xgAQrvkxt3WZi0AAAAASUVORK5CYII=)  |
| ------ | ------ | ------ |
| Akhil Jain | https://github.com/AkhilJain5 | https://www.linkedin.com/in/akhil-jain-61107122a/ |
| Sachin Rajawat | https://github.com/Sachin-71 | https://www.linkedin.com/in/sachin-rajawat-119ba4237/ |
| Pranav Gupta | https://github.com/pranav412-code |  |
| Sarang Kishor Masurkar  | https://github.com/SarangMasurkar | https://www.linkedin.com/in/sarang-masurkar-643819246/ |
| Shahnaz Ali | https://github.com/S-ali24 |  |


