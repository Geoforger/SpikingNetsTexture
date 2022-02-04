# Import libraries
import nest
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt

# Reset NEST kernal
nest.ResetKernel()  # Reset kernal to prevent errors

# Create network
# Create input layer (n = no. pixels in camera)
input_layer = nest.Create("izhikevich", 240*180)

voltmeter = nest.Create("voltmeter")

# Set constant input
nest.SetStatus(input_layer, "I_e", 100.0)
nest.SetStatus(voltmeter, [{"withgid": True}])

nest.Connect(voltmeter, input_layer)

nest.Simulate(10.0)

nest.voltage_trace.from_device(voltmeter)
