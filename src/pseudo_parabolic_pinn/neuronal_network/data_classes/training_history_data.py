import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from matplotlib.figure import Figure

@dataclass
class TrainingHistoryData():

    r""" Class to save data generated during the training process
    
    Args: 
        list_complete_loss (list[float]): List to store the complete loss.
        list_pde_loss (list[float]): List to store pde loss.
        list_init_loss (list[float]): List to store loss from the initial
            condition.
        list_boundary_loss (list[float]): List to store loss from the boundary
            condition.
        list_validation_max_norm (list[float]): List to store the 
            C([0,T]; C(\overline(\Omega))) norm of the error (u-u_nn) on 
            a validation data set.
        list_validation_l2_norm (list[float]): List to store the 
            C([0,T]; L^2(\Omega))) norm of the error (u-u_nn) on 
            a validation data set. 
        list_validation_h1_norm (list[float]): List to store the 
            C([0,T]; H^1(\Omega))) norm of the error (u-u_nn) on 
            a validation data set.    
    """
    
    complete_loss: list[float] = field(default_factory=list[float])
    pde_loss: list[float] = field(default_factory=list[float])
    init_loss: list[float] = field(default_factory=list[float])
    boundary_loss: list[float] = field(default_factory=list[float])

    list_validation_max_norm: list[float] = field(default_factory=list[float])
    list_validation_l2_norm: list[float] = field(default_factory=list[float])
    list_validation_h1_norm: list[float] = field(default_factory=list[float])

    def update_loss_lists(self, 
                          complete_loss: float,
                          pde_loss: float,
                          init_loss: float,
                          boundary_loss: float) -> None:
        
        r""" Update the loss lists
        
        Args: 
            complete_loss (float): Current complete loss.
            pde_loss (float): Current pde loss.
            init_loss (float): Current init_loss.
            boundary_loss (float): Current boundary loss.
        """

        self.complete_loss.append(complete_loss)
        self.pde_loss.append(pde_loss)
        self.init_loss.append(init_loss)
        self.boundary_loss.append(boundary_loss)

    def update_norms_lists(self,
                           max_norm: float,
                           l2_norm: float,
                           h1_norm: float) -> None:
        
        r""" Update the lists that store the norms
        
        Args:
            max_norm (float): Current maximum norm
            l2_norm (float): Current L^2 norm
            h1_norm (float): Current H^1 norm.
        
        """

        self.list_validation_max_norm.append(max_norm)
        self.list_validation_l2_norm.append(l2_norm)
        self.list_validation_h1_norm.append(h1_norm)

    @property
    def loss_lists(self) -> tuple[list[float], list[float],
                                  list[float], list[float]]:

        r""" Returns lists of the different loss functions
        
        Returns:
            tuple[list[float], list[float], list[float], list[float]]:
                Tuple consisting of:
                    - **list[float]**: The complete loss
                    - **list[float]**: The phyiscal loss
                    - **list[float]**: The intial loss
                    - **list[float]**: The boundary loss
        """

        return self.complete_loss, self.pde_loss, self.init_loss, self.boundary_loss

    def present_loss_graphs(self, 
                            steps:int = 1) -> Figure:

        r""" Based on the stored losses graphs are generated that
        illustrate the losses curves
        
        Args:
            step (int, optional): Downsampling interval for plotting. For example, 
                `step=100` plots every 100th epoch. Defaults to 1.

        Returns:
            Figure: Figure that iluustrates the loss curves.
        
        """

        def partition_loss_lists(loss_lists: list[float]) -> list[float]:

            # Number of chunks
            number_chunks = len(loss_lists) // steps
            chunks = [loss_lists[i*steps: (i+1)*steps] for i in range(0,number_chunks)]
            if len(loss_lists) % steps !=0 :
                chunks.append(loss_lists[number_chunks*steps:])

            mean_chunks = [np.mean(chunks[i], dtype=float) for i in range(len(chunks))]

            return mean_chunks
            

        # Determine number chunks
        complete_chunks = partition_loss_lists(self.complete_loss)
        pde_chunks = partition_loss_lists(self.pde_loss)
        init_chunks = partition_loss_lists(self.init_loss)
        boundary_chunks = partition_loss_lists(self.boundary_loss)

        # Compute numbe of chunks
        number_chunks = len(complete_chunks)
        
        # Create figure
        fig, ax = plt.subplots()

        ax.plot([i*steps for i in range(1,number_chunks+1)], complete_chunks, label="complete")
        ax.plot([i*steps for i in range(1,number_chunks+1)], pde_chunks, label="pde")
        ax.plot([i*steps for i in range(1,number_chunks+1)], init_chunks, label="initial")
        ax.plot([i*steps for i in range(1,number_chunks+1)], boundary_chunks, label="boundary")

        ax.grid(visible=True, which="both")
        ax.legend(loc="best")
        ax.set_facecolor("lightgrey")

        if steps % 10 == 0: 
            ax.set_xscale("log")

        ax.set_yscale("log")
        ax.set_xlabel("Number of chunks")
        ax.set_ylabel("Mean Loss")

        return fig
