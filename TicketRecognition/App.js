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
} from "react-native";
import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";

export default function App() {
  const [image, setImage] = useState(null);
  const [processingMethod, setProcessingMethod] = useState("easyocr"); // "easyocr" o "trocr"
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  // Configuración del servidor (cambiar según tu IP local)
  const SERVER_URL = "http://192.168.1.100:5000"; // Cambiar por tu IP local

  // Solicitar permisos para la cámara
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

  // Solicitar permisos para la galería
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

  // Tomar foto con la cámara
  const takePhoto = async () => {
    const hasPermission = await requestCameraPermission();
    if (!hasPermission) return;

    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null); // Limpiar resultado anterior
    }
  };

  // Seleccionar foto de la galería
  const pickImage = async () => {
    const hasPermission = await requestMediaLibraryPermission();
    if (!hasPermission) return;

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null); // Limpiar resultado anterior
    }
  };

  // Procesar imagen en el servidor
  const processImage = async () => {
    if (!image) {
      Alert.alert("Error", "No hay imagen para procesar");
      return;
    }

    setIsProcessing(true);
    setResult(null);

    try {
      // Convertir imagen a base64
      const response = await fetch(image);
      const blob = await response.blob();
      const reader = new FileReader();

      reader.onloadend = async () => {
        const base64Image = reader.result;

        try {
          // Enviar al servidor
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

          const data = await serverResponse.json();

          if (data.success) {
            setResult(data);
            Alert.alert(
              "✅ Procesamiento exitoso",
              `Total detectado: ${data.trocr_prediction}`
            );
          } else {
            Alert.alert("Error", data.error || "Error al procesar la imagen");
          }
        } catch (error) {
          Alert.alert(
            "Error de conexión",
            `No se pudo conectar al servidor: ${error.message}`
          );
        } finally {
          setIsProcessing(false);
        }
      };

      reader.readAsDataURL(blob);
    } catch (error) {
      Alert.alert("Error", `Error al leer la imagen: ${error.message}`);
      setIsProcessing(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>TicketRecognition</Text>
          <Text style={styles.subtitle}>Captura o selecciona un ticket</Text>
        </View>

        {image ? (
          <View style={styles.imageContainer}>
            <Image source={{ uri: image }} style={styles.image} />
            <TouchableOpacity
              style={styles.clearButton}
              onPress={() => setImage(null)}
            >
              <Text style={styles.clearButtonText}>✕ Limpiar</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.placeholderContainer}>
            <Text style={styles.placeholderText}>📸</Text>
            <Text style={styles.placeholderSubtext}>
              No hay imagen seleccionada
            </Text>
          </View>
        )}

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

        {result && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>📊 Resultados:</Text>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Total detectado:</Text>
              <Text style={styles.resultValue}>{result.trocr_prediction}</Text>
            </View>
            {result.easyocr_detection && (
              <View style={styles.resultItem}>
                <Text style={styles.resultLabel}>EasyOCR detectó:</Text>
                <Text style={styles.resultValue}>
                  {result.easyocr_detection}
                </Text>
              </View>
            )}
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Info:</Text>
              <Text style={styles.resultInfo}>{result.info}</Text>
            </View>
          </View>
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
  },
  placeholderText: {
    fontSize: 80,
    marginBottom: 10,
  },
  placeholderSubtext: {
    fontSize: 16,
    color: "#999",
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
    justifyContent: "space-between",
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
  resultContainer: {
    backgroundColor: "#fff",
    padding: 20,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "#4CAF50",
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#333",
    marginBottom: 15,
  },
  resultItem: {
    marginBottom: 12,
  },
  resultLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#666",
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: "700",
    color: "#4CAF50",
  },
  resultInfo: {
    fontSize: 14,
    color: "#666",
    fontStyle: "italic",
  },
});
