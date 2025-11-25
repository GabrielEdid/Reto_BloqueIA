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
          <View style={styles.infoContainer}>
            <Text style={styles.infoText}>✓ Imagen cargada correctamente</Text>
            <Text style={styles.infoSubtext}>
              Aquí puedes procesar la imagen del ticket
            </Text>
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
  infoContainer: {
    backgroundColor: "#e8f5e9",
    padding: 20,
    borderRadius: 12,
    alignItems: "center",
  },
  infoText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#2e7d32",
    marginBottom: 5,
  },
  infoSubtext: {
    fontSize: 14,
    color: "#558b2f",
  },
});
