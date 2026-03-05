# Input Data Format Instructions

This section outlines the detailed input data format required for the code published in `pinn_PVLoop.py`.

## Input Data Structure

The input data should be structured as follows:

1. **Parameter 1**: Description of this parameter, including its expected type (e.g., integer, float) and valid range.
2. **Parameter 2**: Description of this parameter, including its expected type (e.g., string, array) and valid values.
3. **Parameter 3**: Description of this parameter with additional constraints or specifications.

## Example Input Data

An exemplar JSON format representing the input data:

```json
{
  "parameter1": value,
  "parameter2": ["value1", "value2"],
  "parameter3": {
    "subParameter1": value,
    "subParameter2": "value"
  }
}
```

Ensure all input data adheres to the specified formats and constraints for successful execution of the `pinn_PVLoop.py` code.