
from func import *
noisy_image = cv2.imread('./noisy_cube.jpg', cv2.IMREAD_GRAYSCALE) / 255

PATCH_SIZE = 5
REDUCED_PATCH_SIZE = 10
NLM_RANGE_STD = 0.05


def nlm(image, patch_size, reduced_patch_size, sigma_r, num_neighbors=10):
    radius = patch_size // 2
    height, width = image.shape

    padded_image = np.pad(image, radius, mode='reflect')
    patches = np.zeros((height * width, patch_size ** 2,))
    for i in range(radius, height + radius):
        for j in range(radius, width + radius):
            patch = padded_image[i - radius:i + radius + 1, j - radius:j + radius + 1]
            patches[(i - radius) * width + (j - radius), :] = patch.flatten()

    transformed_patches = sklearn.decomposition.PCA(n_components=reduced_patch_size).fit_transform(patches)
    tree = sklearn.neighbors.BallTree(transformed_patches, leaf_size=2)
    output = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            patch_index = i * width + j
            patch = patches[patch_index]
            representative_patch = np.expand_dims(transformed_patches[patch_index], 0)
            _, neighbor_indices = tree.query(representative_patch, k=num_neighbors)
            neighbor_indices = neighbor_indices[0, 1:]
            pixel_indices = np.array([
                (neighbor_index // width, neighbor_index % width)
                for neighbor_index in neighbor_indices
            ])

            pixels = image[pixel_indices[:, 0], pixel_indices[:, 1]]
            weights = np.array([
                gaussian_distance(patch, patches[neighbor_index], sigma_r)
                for neighbor_index in neighbor_indices
            ])

            output[i, j] = np.sum(weights * pixels) / np.sum(weights)

    return output


nlm_image = nlm(noisy_image, PATCH_SIZE, REDUCED_PATCH_SIZE, NLM_RANGE_STD)

plt.imshow(nlm_image, cmap='gray')
plt.show()