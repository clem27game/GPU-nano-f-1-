# 🚀 GPU Nano F-1 - Documentation Complète

## 📋 Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Caractéristiques](#caractéristiques)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contribution](#contribution)
- [Licence](#licence)

---

## 👁️ Vue d'ensemble

**GPU Nano F-1** est une unité de traitement graphique compacte et performante conçue pour les applications embarquées et les environnements aux ressources limitées. Avec ses 10 Go de VRAM, ce GPU offre un excellent rapport performances/taille pour les tâches de calcul intensif, le machine learning et le traitement vidéo.

### Points clés
✨ **Compact** : Format nano idéal pour l'intégration  
⚡ **Performant** : 10 Go de VRAM dédiée  
🔧 **Configurable** : Options d'optimisation multiples  
🎯 **Spécialisé** : Conçu pour les charges intensives

---

## 🎯 Caractéristiques

| Caractéristique | Détail |
|---|---|
| **Mémoire** | 10 Go GDDR6 |
| **Type** | GPU Nano |
| **Modèle** | F-1 |
| **Langage principal** | C |
| **Architecture** | Optimisée pour le calcul parallèle |
| **Consommation** | Efficace énergétiquement |

### Capacités
- Calcul CUDA/GPU généraliste
- Traitement parallèle haute performance
- Support des opérations matricielles
- Optimisation mémoire avancée

---

## 📦 Installation

### Prérequis
- Système Linux/Windows/macOS compatible
- Compilateur C (GCC/Clang)
- NVIDIA CUDA Toolkit (si applicable)
- Pilotes GPU à jour

### Étapes d'installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/clem27game/GPU-nano-f-1-.git
cd GPU-nano-f-1-

# 2. Créer le répertoire de build
mkdir build
cd build

# 3. Compiler
cmake ..
make

# 4. Installer (optionnel)
sudo make install
```

### Vérification de l'installation
```bash
./gpu-nano-f1 --version
./gpu-nano-f1 --check-hardware
```

---

## 🛠️ Utilisation

### Exemple basique

```c
#include "gpu_nano_f1.h"

int main() {
    // Initialiser le GPU
    gpu_context_t ctx = gpu_init();
    
    // Allouer de la mémoire
    float* device_data = gpu_malloc(ctx, 1024 * sizeof(float));
    
    // Exécuter des opérations
    gpu_compute(ctx, device_data, 1024);
    
    // Libérer les ressources
    gpu_free(ctx, device_data);
    gpu_shutdown(ctx);
    
    return 0;
}
```

### Compilation d'application
```bash
gcc -o my_app my_app.c -lgpu_nano_f1
```

---

### Composants principaux

#### 1. **Core Module**
Gestion des contextes GPU et des opérations fondamentales

#### 2. **Memory Module**
Allocation, libération et transfert de données

#### 3. **Compute Module**
Kernels GPU et exécution de tâches parallèles

#### 4. **Driver Layer**
Interface avec le matériel spécifique

---

## ⚙️ Configuration

### Variables d'environnement

```bash
# Définir le nombre de threads GPU
export GPU_NANO_MAX_THREADS=2048

# Activer le mode debug
export GPU_NANO_DEBUG=1

# Définir le niveau d'optimisation
export GPU_NANO_OPTIMIZATION_LEVEL=3
```

### Fichier de configuration (optionnel)

Créez `gpu_config.ini` :

```ini
[memory]
max_allocation = 10737418240  # 10 Go en bytes
allocation_strategy = dynamic

[performance]
optimization_level = 3
max_threads = 2048
cache_size = 262144

[debug]
enable_logging = true
log_level = info
```

---

## 🔍 Troubleshooting

### Problème : GPU non détecté

```bash
# Vérifier les pilotes
nvidia-smi

# Réinitialiser le GPU
./gpu-nano-f1 --reset
```

### Problème : Erreur d'allocation mémoire

- Vérifier la mémoire disponible
- Réduire la taille des allocations
- Vérifier les fuites mémoire

### Problème : Performance faible

- Vérifier l'utilisation du GPU : `gpu_nano_f1 --stats`
- Optimiser la configuration
- Vérifier la taille des kernels

### Logs et diagnostics

```bash
# Activer les logs détaillés
export GPU_NANO_DEBUG=1
./my_app

# Générer un rapport de diagnostic
./gpu-nano-f1 --diagnostic-report
```

---

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :

1. **Fork** le dépôt
2. **Créer une branche** : `git checkout -b feature/ma-feature`
3. **Commit** vos changements : `git commit -m 'Ajout de ma-feature'`
4. **Push** vers la branche : `git push origin feature/ma-feature`
5. **Ouvrir une Pull Request**

### Directives
- Respecter le style de code C
- Ajouter des tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation
- Signaler les bugs via les Issues

---

## 📚 Ressources supplémentaires

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [GPU Computing Fundamentals](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Forum Communautaire](https://github.com/clem27game/GPU-nano-f-1-/discussions)

---

## 📝 Licence

MIT 
---

## ✉️ Contact & Support

- **Auteur** : clem27game
- **Issues** : [GitHub Issues](https://github.com/clem27game/GPU-nano-f-1-/issues)
- **Discussions** : [GitHub Discussions](https://github.com/clem27game/GPU-nano-f-1-/discussions)

---

# Utilisation (exemple de codes) :

Install :
!sed -i '/#include <math.h>/a #include <limits.h>' main.c
!sed -i 's/#define GPU_VRAM_SIZE (10 \* 1024 \* 1024 \* 1024)/#define GPU_VRAM_SIZE (10ULL * 1024 * 1024 * 1024)/' main.c
!gcc -fPIC -shared main.c -o libgpu_nano_f1.so -fopenmp
print('--- Diagnostics ---')
!ldd libgpu_nano_f1.so
!nm -D libgpu_nano_f1.so | grep omp_unset_lock
!nm -D /lib/x86_64-linux-gnu/libgomp.so.1 | grep omp_unset_lock
print('fin')

Export :
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/ && python gpu_nano_f1.py
print('fin'

Exemple :
import numpy as np

# Assuming GPUNanoF1 class is defined in gpu_nano_f1.py and imported
# (If not already imported, you might need to add: from gpu_nano_f1 import GPUNanoF1)

# For this example, let's instantiate the class assuming it's available in the environment
# Since we are in the same directory, we can directly import it after the previous steps.
# If `gpu_nano_f1.py` was executed as a script, its definitions might not be in scope.
# Let's ensure the class is imported for clarity.
from gpu_nano_f1 import GPUNanoF1

gpu = GPUNanoF1()

# Create a 2x3 tensor
rows_a, cols_a = 2, 3
tensor_a_id = gpu.create_tensor(rows_a, cols_a)
print(f"Created Tensor A with ID: {tensor_a_id}")

# Set data for Tensor A
data_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
gpu.set_data(tensor_a_id, data_a)
print(f"Set data for Tensor A:\n{data_a}")

# Get data back from Tensor A to verify
retrieved_data_a = gpu.get_data(tensor_a_id, (rows_a, cols_a))
print(f"Retrieved data for Tensor A:\n{retrieved_data_a}")

# Example of memory info
used_mem, total_mem = gpu.memory_info()
print(f"GPU Memory Used: {used_mem:.2f} GB / Total: {total_mem:.2f} GB")

# Don't forget to shutdown the GPU context when done
# gpu.shutdown()
print('fin')

Résultat :

Created Tensor A with ID: 1
Set data for Tensor A:
[[1. 2. 3.]
 [4. 5. 6.]]
Retrieved data for Tensor A:
[[1. 2. 3.]
 [4. 5. 6.]]
GPU Memory Used: 0.00 GB / Total: 10.00 GB
fin
**Dernière mise à jour** : Mars 2026
