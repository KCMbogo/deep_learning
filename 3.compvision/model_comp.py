import pandas as pd
import matplotlib.pyplot as plt
from model_0 import model_0_results, total_train_time_model_0
from model_1 import model_1_results, total_train_time_model_1
from model_2 import model_2_results, total_train_time_model_2

compare_results = pd.DataFrame([model_0_results,
                                model_1_results,
                                model_2_results])

compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]

print("\nModel results:\n", compare_results)

print(compare_results.columns)

compare_results.set_index("model_name")["model_loss"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.show()



