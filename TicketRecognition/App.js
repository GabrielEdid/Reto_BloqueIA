import React, { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  Platform,
  Modal,
  ActivityIndicator,
  Dimensions,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

export default function App() {
  const [image, setImage] = useState(null);
  const [processingMethod, setProcessingMethod] = useState("easyocr"); // "easyocr" | "trocr" | "doctr"
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [showResultModal, setShowResultModal] = useState(false);

  // Configuración del servidor (cambiar según tu IP local)
  const SERVER_URL = "http://10.49.126.247:8000";

  const requestCameraPermission = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permiso denegado",
        "Se necesita acceso a la cámara para tomar fotos."
      );
      return false;
    }
    return true;
  };

  const requestMediaLibraryPermission = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permiso denegado",
        "Se necesita acceso a la galería para seleccionar fotos."
      );
      return false;
    }
    return true;
  };

  const takePhoto = async () => {
    const hasPermission = await requestCameraPermission();
    if (!hasPermission) return;

    const resultPick = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!resultPick.canceled) {
      setImage(resultPick.assets[0].uri);
      setResult(null);
      setShowResultModal(false);
    }
  };

  const pickImage = async () => {
    const hasPermission = await requestMediaLibraryPermission();
    if (!hasPermission) return;

    const resultPick = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!resultPick.canceled) {
      setImage(resultPick.assets[0].uri);
      setResult(null);
      setShowResultModal(false);
    }
  };

  const uriToBase64 = async (uri) => {
    const response = await fetch(uri);
    const blob = await response.blob();

    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(blob);
    });
  };

  const processImage = async () => {
    if (!image) {
      Alert.alert("Error", "No hay imagen para procesar");
      return;
    }

    setIsProcessing(true);
    setResult(null);
    setShowResultModal(false);

    try {
      const base64Image = await uriToBase64(image);

      const serverResponse = await fetch(`${SERVER_URL}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image,
          method: processingMethod,
        }),
      });

      console.log("Respuesta del servidor (status):", serverResponse.status);

      if (!serverResponse.ok) {
        const infoMensaje = `Respuesta inválida del servidor (${serverResponse.status})`;
        const finalResult = {
          method: processingMethod,
          trocr_prediction: null,
          easyocr_detection: null,
          info: infoMensaje,
          error: true,
        };
        console.log("Resultado procesado (error status):", finalResult);
        setResult(finalResult);
        setShowResultModal(true);
        return;
      }

      let data = null;
      try {
        data = await serverResponse.json();
      } catch (e) {
        console.log("Error parseando JSON:", e);
      }

      // Si ni siquiera hay JSON válido
      if (!data) {
        const infoMensaje =
          "No se pudo interpretar la respuesta del servidor. Intenta nuevamente.";
        const finalResult = {
          method: processingMethod,
          trocr_prediction: null,
          easyocr_detection: null,
          info: infoMensaje,
          error: true,
        };
        console.log("Resultado procesado (sin JSON):", finalResult);
        setResult(finalResult);
        setShowResultModal(true);
        return;
      }

      // success = false desde el servidor
      if (data.success === false) {
        const infoMensaje =
          data.error || "No se pudo procesar la imagen en el servidor.";
        const finalResult = {
          method: data.method || processingMethod,
          trocr_prediction: null,
          easyocr_detection: null,
          info: infoMensaje,
          error: true,
        };
        console.log("Resultado procesado (success=false):", finalResult);
        setResult(finalResult);
        setShowResultModal(true);
        return;
      }

      // En este punto success es true, pero puede no haber predicciones
      let finalResult = {
        ...data,
        method: data.method || processingMethod,
      };

      // Si está todo vacío, construimos un mensaje de "ningún modelo devolvió total"
      if (
        !finalResult.trocr_prediction &&
        !finalResult.easyocr_detection &&
        !finalResult.info
      ) {
        let infoMensaje =
          "Ninguno de los modelos logró detectar un total en esta imagen. Intenta con otra foto o ajusta el encuadre.";

        if (processingMethod === "trocr") {
          infoMensaje =
            "El modelo de Solo TrOCR no logró detectar un total en esta imagen. Prueba con EasyOCR + TrOCR o intenta otra foto.";
        } else if (processingMethod === "easyocr") {
          infoMensaje =
            "EasyOCR + TrOCR no lograron detectar un total en esta imagen. Intenta tomar la foto más cerca del total o con mejor iluminación.";
        } else if (processingMethod === "doctr") {
          infoMensaje =
            "docTR no logró detectar un total en esta imagen. Intenta otra foto o un encuadre más directo al total.";
        }

        finalResult = {
          ...finalResult,
          trocr_prediction: null,
          easyocr_detection: null,
          info: infoMensaje,
          error: true,
        };
      }

      console.log("Resultado procesado (final):", finalResult);
      setResult(finalResult);
      setShowResultModal(true);
    } catch (error) {
      console.log("Error en processImage:", error);

      // Siempre construimos un resultado para mostrar en el modal
      const infoMensaje = `Ocurrió un error al procesar la imagen. Intenta nuevamente.\n\nDetalle técnico: ${
        error && error.message ? error.message : String(error)
      }`;

      const finalResult = {
        method: processingMethod,
        trocr_prediction: null,
        easyocr_detection: null,
        info: infoMensaje,
        error: true,
      };

      setResult(finalResult);
      setShowResultModal(true);

      // Si quieres, puedes dejar también el Alert:
      // Alert.alert("Error", "Ocurrió un error al procesar la imagen. Intenta nuevamente.");
    } finally {
      setIsProcessing(false);
    }
  };

  const clearImage = () => {
    setImage(null);
    setResult(null);
    setShowResultModal(false);
  };

  const closeResultModal = () => {
    setShowResultModal(false);
  };

  const getMethodLabel = (method) => {
    if (method === "easyocr") return "EasyOCR + TrOCR";
    if (method === "trocr") return "Solo TrOCR";
    return "docTR";
  };

  const getTotalText = (res) => {
    // Si hay predicción numérica/string, la mostramos directamente
    if (res.trocr_prediction) {
      return res.trocr_prediction;
    }

    // Mensajes específicos por modelo cuando no se obtuvo predicción
    if (res.method === "easyocr") {
      return "EasyOCR + TrOCR no lograron detectar un total en esta imagen.";
    }

    if (res.method === "trocr") {
      return "Solo TrOCR no logró detectar un total en esta imagen.";
    }

    if (res.method === "doctr") {
      return "docTR no logró detectar un total en esta imagen.";
    }

    // Fallback genérico
    return "No se detectó un total en esta imagen.";
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />

      {/* Modal de procesamiento */}
      <Modal visible={isProcessing} transparent animationType="fade">
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Procesando imagen...</Text>
            <Text style={styles.loadingSubtext}>
              Esto puede tomar unos segundos
            </Text>
          </View>
        </View>
      </Modal>

      {/* Modal de resultados */}
      <Modal
        visible={showResultModal}
        transparent
        animationType="slide"
        onRequestClose={closeResultModal}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContainer}>
            <ScrollView
              contentContainerStyle={styles.modalScrollContent}
              showsVerticalScrollIndicator
            >
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>📊 Resultados</Text>
                <TouchableOpacity
                  style={styles.closeButton}
                  onPress={closeResultModal}
                >
                  <Text style={styles.closeButtonText}>✕</Text>
                </TouchableOpacity>
              </View>

              {image && (
                <View style={styles.modalImageContainer}>
                  <Image
                    source={{ uri: image }}
                    style={styles.modalImage}
                    resizeMode="contain"
                  />
                </View>
              )}

              {result && (
                <View style={styles.modalResultsContainer}>
                  <View style={styles.modalResultItem}>
                    <Text style={styles.modalResultLabel}>
                      💰 Total Detectado:
                    </Text>
                    <Text style={styles.modalResultValue}>
                      {getTotalText(result)}
                    </Text>
                  </View>

                  {result.easyocr_detection ? (
                    <View style={styles.modalResultItem}>
                      <Text style={styles.modalResultLabel}>
                        🔍 EasyOCR Detectó:
                      </Text>
                      <Text style={styles.modalResultSecondary}>
                        {result.easyocr_detection}
                      </Text>
                    </View>
                  ) : null}

                  <View style={styles.modalResultItem}>
                    <Text style={styles.modalResultLabel}>📝 Método:</Text>
                    <Text style={styles.modalResultSecondary}>
                      {getMethodLabel(result.method)}
                    </Text>
                  </View>

                  {result.info && (
                    <View style={styles.modalResultItem}>
                      <Text style={styles.modalResultLabel}>
                        ℹ️ Información:
                      </Text>
                      <Text style={styles.modalResultInfo}>{result.info}</Text>
                    </View>
                  )}
                </View>
              )}

              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={closeResultModal}
              >
                <Text style={styles.modalCloseButtonText}>Cerrar</Text>
              </TouchableOpacity>
            </ScrollView>
          </View>
        </View>
      </Modal>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>OCR Tickets</Text>
          <Text style={styles.subtitle}>
            Toma o selecciona una foto para detectar el total del ticket
          </Text>
        </View>

        <View style={styles.imageContainer}>
          {image ? (
            <>
              <Image
                source={{ uri: image }}
                style={styles.image}
                resizeMode="contain"
              />
              <TouchableOpacity style={styles.clearButton} onPress={clearImage}>
                <Text style={styles.clearButtonText}>Borrar imagen</Text>
              </TouchableOpacity>
            </>
          ) : (
            <View style={styles.placeholderContainer}>
              <Text style={styles.placeholderText}>🧾</Text>
              <Text style={styles.placeholderSubtext}>
                No hay imagen seleccionada
              </Text>
            </View>
          )}
        </View>

        <View style={styles.methodContainer}>
          <Text style={styles.methodTitle}>Método de Procesamiento:</Text>

          <TouchableOpacity
            style={styles.checkboxContainer}
            onPress={() => setProcessingMethod("easyocr")}
          >
            <View style={styles.checkbox}>
              {processingMethod === "easyocr" && (
                <View style={styles.checkboxChecked} />
              )}
            </View>
            <Text style={styles.checkboxLabel}>EasyOCR + TrOCR</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.checkboxContainer}
            onPress={() => setProcessingMethod("trocr")}
          >
            <View style={styles.checkbox}>
              {processingMethod === "trocr" && (
                <View style={styles.checkboxChecked} />
              )}
            </View>
            <Text style={styles.checkboxLabel}>Solo TrOCR</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.checkboxContainer}
            onPress={() => setProcessingMethod("doctr")}
          >
            <View style={styles.checkbox}>
              {processingMethod === "doctr" && (
                <View style={styles.checkboxChecked} />
              )}
            </View>
            <Text style={styles.checkboxLabel}>docTR</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={takePhoto}>
            <Text style={styles.buttonIcon}>📷</Text>
            <Text style={styles.buttonText}>Tomar Foto</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonIcon}>🖼️</Text>
            <Text style={styles.buttonText}>Seleccionar Foto</Text>
          </TouchableOpacity>
        </View>

        {image && (
          <TouchableOpacity
            style={[
              styles.processButton,
              isProcessing && styles.processButtonDisabled,
            ]}
            onPress={processImage}
            disabled={isProcessing}
          >
            <Text style={styles.processButtonText}>
              {isProcessing ? "⏳ Procesando..." : "🔍 Procesar Ticket"}
            </Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  scrollContent: {
    flexGrow: 1,
    paddingTop: Platform.OS === "ios" ? 60 : 40,
    paddingBottom: 40,
    paddingHorizontal: 20,
  },
  header: {
    alignItems: "center",
    marginBottom: 30,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: "#666",
    textAlign: "center",
  },
  imageContainer: {
    alignItems: "center",
    marginBottom: 30,
  },
  image: {
    width: "100%",
    height: 400,
    borderRadius: 12,
    marginBottom: 15,
  },
  clearButton: {
    backgroundColor: "#ff4444",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  clearButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  placeholderContainer: {
    alignItems: "center",
    justifyContent: "center",
    height: 300,
    backgroundColor: "#fff",
    borderRadius: 12,
    marginBottom: 30,
    borderWidth: 2,
    borderColor: "#ddd",
    borderStyle: "dashed",
    paddingHorizontal: 20,
  },
  placeholderText: {
    fontSize: 80,
    marginBottom: 10,
  },
  placeholderSubtext: {
    fontSize: 16,
    color: "#999",
    textAlign: "center",
  },
  methodContainer: {
    backgroundColor: "#fff",
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  methodTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
    marginBottom: 15,
  },
  checkboxContainer: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 10,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "#007AFF",
    marginRight: 12,
    justifyContent: "center",
    alignItems: "center",
  },
  checkboxChecked: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: "#007AFF",
  },
  checkboxLabel: {
    fontSize: 16,
    color: "#333",
  },
  buttonContainer: {
    flexDirection: "row",
    marginBottom: 20,
  },
  button: {
    flex: 1,
    backgroundColor: "#007AFF",
    paddingVertical: 20,
    paddingHorizontal: 15,
    borderRadius: 12,
    alignItems: "center",
    marginHorizontal: 5,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  buttonIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
    textAlign: "center",
  },
  processButton: {
    backgroundColor: "#4CAF50",
    paddingVertical: 18,
    paddingHorizontal: 30,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 20,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  processButtonDisabled: {
    backgroundColor: "#9E9E9E",
  },
  processButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
  loadingOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    justifyContent: "center",
    alignItems: "center",
  },
  loadingContainer: {
    backgroundColor: "#fff",
    borderRadius: 20,
    padding: 40,
    alignItems: "center",
    minWidth: 200,
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
  },
  loadingText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
    marginTop: 20,
  },
  loadingSubtext: {
    fontSize: 14,
    color: "#666",
    marginTop: 8,
    textAlign: "center",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalContainer: {
    backgroundColor: "#fff",
    borderRadius: 20,
    width: SCREEN_WIDTH * 0.95,
    maxHeight: SCREEN_HEIGHT * 0.9,
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
  },
  modalScrollContent: {
    padding: 20,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 2,
    borderBottomColor: "#f0f0f0",
  },
  modalTitle: {
    fontSize: 26,
    fontWeight: "bold",
    color: "#333",
  },
  closeButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#ff4444",
    justifyContent: "center",
    alignItems: "center",
  },
  closeButtonText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
  },
  modalImageContainer: {
    alignItems: "center",
    marginBottom: 20,
    backgroundColor: "#f9f9f9",
    borderRadius: 12,
    padding: 10,
  },
  modalImage: {
    width: "100%",
    height: 400,
    borderRadius: 12,
  },
  modalResultsContainer: {
    backgroundColor: "#f9f9f9",
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  modalResultItem: {
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
  },
  modalResultLabel: {
    fontSize: 16,
    fontWeight: "600",
    color: "#555",
    marginBottom: 8,
  },
  modalResultValue: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#4CAF50",
  },
  modalResultSecondary: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
  },
  modalResultInfo: {
    fontSize: 15,
    color: "#666",
    fontStyle: "italic",
    lineHeight: 22,
  },
  modalCloseButton: {
    backgroundColor: "#007AFF",
    paddingVertical: 16,
    paddingHorizontal: 30,
    borderRadius: 12,
    alignItems: "center",
    marginTop: 10,
  },
  modalCloseButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
  },
});
