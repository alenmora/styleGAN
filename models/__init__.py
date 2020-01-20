
def toggle_grad(model, requires_grad):
    """
    Function to change the trainability
    of a model
    """
    for p in model.parameters():
        p.requires_grad_(requires_grad)