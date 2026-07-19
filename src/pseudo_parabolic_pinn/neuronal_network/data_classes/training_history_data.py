from dataclasses import dataclass, field
import matplotlib.pyplot as plt

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
    
    list_complete_loss: list[float] = field(default_factory=list[float])
    list_pde_loss: list[float] = field(default_factory=list[float])
    list_init_loss: list[float] = field(default_factory=list[float])
    list_boundary_loss: list[float] = field(default_factory=list[float])

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

        self.list_complete_loss.append(complete_loss)
        self.list_pde_loss.append(pde_loss)
        self.list_init_loss.append(init_loss)
        self.list_boundary_loss.append(boundary_loss)

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

    def present_loss_graphs(self) -> Figure:

        r""" Based on the stored losses graphs are generated that
        illustrate the losses curves
        
        Returns:
            Figure: Figure that iluustrates the loss curves.
        
        """
        
        # Determine number of epochs
        epochs = len(self.list_complete_loss)

        # Create figure
        fig, ax = plt.subplots()

        ax.plot([i for i in range(1,epochs+1)], self.list_complete_loss, label="complete")
        ax.plot([i for i in range(1,epochs+1)], self.list_pde_loss, label="pde")
        ax.plot([i for i in range(1,epochs+1)], self.list_boundary_loss, label="boundary")
        ax.plot([i for i in range(1,epochs+1)], self.list_init_loss, label="initial")

        ax.grid(visible=True, which="both")
        ax.legend(loc="best")
        ax.set_facecolor("lightgrey")
        #ax.set_xscale("log")
        ax.set_yscale("log")

        return fig
