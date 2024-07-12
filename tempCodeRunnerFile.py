img=X_test[0]
plt.imshow(img, cmap='gray')
# plt.title(f"Predicted Digit: {predicted_digit}, Probability: {probability:.2f}")
plt.axis('off')
plt.show()
Resize to match model input shape
