import numpy as np
import matplotlib.pyplot as plt


def plot_results(estimators, xy_train, shape):
    H, W, _ = shape
    est_r, est_g, est_b = estimators
    r_pred = est_r.predict(xy_train).reshape(H, W)
    g_pred = est_g.predict(xy_train).reshape(H, W)
    b_pred = est_b.predict(xy_train).reshape(H, W)

    rgb_pred = np.stack([r_pred, g_pred, b_pred], axis=-1)
    rgb_pred = np.nan_to_num(rgb_pred, nan=0.0)
    rgb_pred = np.clip(rgb_pred, 0, 255)
    r_pred /= np.max(r_pred)
    g_pred /= np.max(g_pred)
    b_pred /= np.max(b_pred)
    rgb_pred /= np.max(rgb_pred)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(rgb_pred)
    plt.title('Approximated Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(r_pred, cmap='Reds')
    plt.title('Approximated Red Channel')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(g_pred, cmap='Greens')
    plt.title('Approximated Green Channel')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(b_pred, cmap='Blues')
    plt.title('Approximated Blue Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_fitness(estimators, gen_start, loss):
    est_r, est_g, est_b = estimators
    fitness1 = est_r.run_details_["best_fitness"][gen_start:]
    fitness2 = est_g.run_details_["best_fitness"][gen_start:]
    fitness3 = est_b.run_details_["best_fitness"][gen_start:]

    # Plot all three
    plt.figure(figsize=(10, 6))
    plt.plot(fitness1, label='Estimator 1', color="red")
    plt.plot(fitness2, label='Estimator 2', color="green")
    plt.plot(fitness3, label='Estimator 3', color="blue")

    plt.title(f"Best Fitness per Generation {loss}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()