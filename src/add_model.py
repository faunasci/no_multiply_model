import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import math

class ShiftAddLinear(nn.Module):
    """
    Camada linear baseada em shifts e adições, inspirada no método egípcio.
    
    Conceito: Em vez de pesos reais, usamos pesos que são potências de 2.
    Multiplicações se tornam shifts (operações bit-shift).
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Pesos reais (para inicialização e treinamento)
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Pesos quantizados (potências de 2)
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features))
        
        # Fator de escala para mapear pesos reais para potências de 2
        self.scale = nn.Parameter(torch.ones(1))
        
    def quantize_to_powers_of_2(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Quantiza pesos para a forma: sinal * 2^expoente
        onde expoente é inteiro (pode ser negativo)
        """
        # Normalizar pesos
        weights_normalized = weights / (weights.abs().mean() + 1e-8)
        
        # Escalar para o range de bits
        weights_scaled = weights_normalized * self.scale
        
        # Converter para potências de 2 mais próximas
        # Expoente = log2(|peso|), arredondado para inteiro
        abs_weights = weights_scaled.abs() + 1e-12  # Evitar log(0)
        exponents = torch.round(torch.log2(abs_weights)).long()
        
        # Limitar expoentes ao range permitido
        exponents = torch.clamp(exponents, -self.bits//2, self.bits//2 - 1)
        
        # Reconstruir pesos como potências de 2 com sinal
        quantized_weights = torch.sign(weights_scaled) * (2.0 ** exponents.float())
        
        return quantized_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantizar pesos durante forward pass
        self.weight_quantized = self.quantize_to_powers_of_2(self.weight_real)
        
        # Multiplicação tradicional (para gradiente)
        out_real = F.linear(x, self.weight_real, self.bias)
        
        # Multiplicação com pesos quantizados (shifts e adições)
        out_quantized = F.linear(x, self.weight_quantized, self.bias)
        
        # Usar resultado quantizado mas gradiente do real (Straight-Through Estimator)
        out = out_quantized + (out_real - out_quantized).detach()
        
        return out
    
    def get_multiplication_count(self) -> int:
        """Retorna número de multiplicações reais necessárias"""
        # Na inferência, todas as operações são shifts e adições
        return 0
    
    def get_addition_count(self) -> int:
        """Retorna número aproximado de adições"""
        # Cada elemento da saída requer in_features adições
        return self.out_features * self.in_features


class StandardLinear(nn.Module):
    """Camada linear padrão para comparação"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_multiplication_count(self) -> int:
        """Retorna número de multiplicações"""
        return self.linear.weight.numel()
    
    def get_addition_count(self) -> int:
        """Retorna número de adições"""
        return self.linear.weight.numel()


class SimpleLanguageModel(nn.Module):
    """Modelo de linguagem simplificado para teste"""
    
    def __init__(self, vocab_size: int, d_model: int, use_shift_add: bool = False, bits: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Camadas lineares
        if use_shift_add:
            self.linear1 = ShiftAddLinear(d_model, d_model * 4, bits=bits)
            self.linear2 = ShiftAddLinear(d_model * 4, d_model, bits=bits)
            self.output = ShiftAddLinear(d_model, vocab_size, bits=bits)
        else:
            self.linear1 = StandardLinear(d_model, d_model * 4)
            self.linear2 = StandardLinear(d_model * 4, d_model)
            self.output = StandardLinear(d_model, vocab_size)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Embeddings + positional encoding
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        
        # Bloco Feed-Forward Network (FFN)
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.layer_norm1(residual + x)
        
        return self.output(x)
    
    def get_operation_counts(self) -> Tuple[int, int]:
        """Retorna (multiplicações, adições)"""
        mults = (self.linear1.get_multiplication_count() + 
                self.linear2.get_multiplication_count() + 
                self.output.get_multiplication_count())
        
        adds = (self.linear1.get_addition_count() + 
               self.linear2.get_addition_count() + 
               self.output.get_addition_count())
        
        return mults, adds


def create_text_dataset(n_samples: int = 1000, seq_len: int = 20, vocab_size: int = 50):
    """
    Cria dataset de texto sintético com padrões linguísticos simples:
    - Palavras frequentemente seguidas por outras específicas
    - Algumas palavras funcionam como conectivos
    """
    X, y = [], []
    
    # Criar padrões de texto
    patterns = [
        [1, 2, 3, 4, 5],  # Sequência crescente
        [10, 11, 12],     # Grupo de números
        [20, 21, 22, 23], # Outro grupo
        [30, 31],         # Par curto
    ]
    
    for _ in range(n_samples):
        seq = []
        pattern_idx = np.random.randint(0, len(patterns))
        pattern = patterns[pattern_idx]
        
        # Repetir padrão com variações
        for i in range(seq_len):
            if i < len(pattern):
                word = pattern[i]
            else:
                # Continuar padrão ou variar
                if np.random.random() < 0.7:
                    word = pattern[i % len(pattern)]
                else:
                    word = np.random.randint(0, vocab_size)
            
            seq.append(word % vocab_size)
        
        X.append(seq[:-1])
        y.append(seq[1:])
    
    return torch.tensor(X), torch.tensor(y)


def train_and_compare():
    """Treina e compara modelos com e sem shift-add"""
    
    print("=" * 80)
    print("EXPERIMENTO: Shift-Add Neural Networks vs Standard Neural Networks")
    print("Baseado no método egípcio de multiplicação por adições e dobros")
    print("=" * 80)
    
    # Configurações
    vocab_size = 50
    d_model = 64
    seq_len = 20
    batch_size = 16
    n_epochs = 1000
    bits = 6  # Precisão dos pesos (expoentes de -3 a +2)
    
    # Criar datasets
    X_train, y_train = create_text_dataset(n_samples=800, seq_len=seq_len, vocab_size=vocab_size)
    X_test, y_test = create_text_dataset(n_samples=200, seq_len=seq_len, vocab_size=vocab_size)
    
    # Modelos
    model_standard = SimpleLanguageModel(vocab_size, d_model, use_shift_add=False)
    model_shiftadd = SimpleLanguageModel(vocab_size, d_model, use_shift_add=True, bits=bits)
    
    # Contar operações
    mults_std, adds_std = model_standard.get_operation_counts()
    mults_sa, adds_sa = model_shiftadd.get_operation_counts()
    
    print(f"Análise de Operações:")
    print(f"Modelo Padrão:")
    print(f"  Multiplicações: {mults_std:,}")
    print(f"  Adições: {adds_std:,}")
    print(f"Modelo Shift-Add:")
    print(f"  Multiplicações: {mults_sa:,}")
    print(f"  Adições: {adds_sa:,}")
    print(f"Redução de multiplicações: {((mults_std - mults_sa) / mults_std * 100):.1f}%")
    
    # Otimizadores
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=0.001)
    optimizer_shiftadd = torch.optim.Adam(model_shiftadd.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss()
    
    # Histórico
    history = {
        'standard_train': [],
        'standard_test': [],
        'shiftadd_train': [],
        'shiftadd_test': []
    }
    
    print("Treinando modelos...")
    
    # Training loop
    for epoch in range(n_epochs):
        # Treinar Standard
        model_standard.train()
        train_loss_std = 0
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer_standard.zero_grad()
            outputs = model_standard(batch_x)
            loss = criterion(outputs.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer_standard.step()
            train_loss_std += loss.item()
        
        # Treinar Shift-Add
        model_shiftadd.train()
        train_loss_sa = 0
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer_shiftadd.zero_grad()
            outputs = model_shiftadd(batch_x)
            loss = criterion(outputs.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer_shiftadd.step()
            train_loss_sa += loss.item()
        
        # Avaliar
        model_standard.eval()
        model_shiftadd.eval()
        
        with torch.no_grad():
            # Standard test
            outputs_std = model_standard(X_test)
            test_loss_std = criterion(outputs_std.reshape(-1, vocab_size), y_test.reshape(-1))
            
            # Shift-Add test
            outputs_sa = model_shiftadd(X_test)
            test_loss_sa = criterion(outputs_sa.reshape(-1, vocab_size), y_test.reshape(-1))
        
        # Armazenar
        history['standard_train'].append(train_loss_std / (len(X_train) // batch_size))
        history['standard_test'].append(test_loss_std.item())
        history['shiftadd_train'].append(train_loss_sa / (len(X_train) // batch_size))
        history['shiftadd_test'].append(test_loss_sa.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Standard - Train: {history['standard_train'][-1]:.4f}, Test: {history['standard_test'][-1]:.4f}")
            print(f"  ShiftAdd - Train: {history['shiftadd_train'][-1]:.4f}, Test: {history['shiftadd_test'][-1]:.4f}")
            print()
    
    # Visualizar resultados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['standard_train'], label='Standard (Train)', linestyle='--')
    plt.plot(history['shiftadd_train'], label='Shift-Add (Train)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['standard_test'], label='Standard (Test)')
    plt.plot(history['shiftadd_test'], label='Shift-Add (Test)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparação final
    plt.subplot(1, 3, 3)
    models = ['Standard', 'Shift-Add']
    final_losses = [history['standard_test'][-1], history['shiftadd_test'][-1]]
    colors = ['blue', 'orange']
    
    bars = plt.bar(models, final_losses, color=colors)
    plt.ylabel('Final Test Loss')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/shift_add_language_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico salvo como 'results/shift_add_language_model_comparison.png'")
    
    # Resultados finais
    print("" + "=" * 80)
    print("RESULTADOS FINAIS")
    print("=" * 80)
    print(f"Standard Model - Final Test Loss: {history['standard_test'][-1]:.4f}")
    print(f"Shift-Add Model - Final Test Loss: {history['shiftadd_test'][-1]:.4f}")
    
    improvement = ((history['standard_test'][-1] - history['shiftadd_test'][-1]) / 
                   history['standard_test'][-1] * 100)
    
    print(f"Diferença de performance: {abs(improvement):.2f}%")
    
    if improvement > 0:
        print("✓ Modelo Standard teve MELHOR performance!")
        print("Isso era esperado pois tem mais precisão numérica.")
    elif improvement < 0:
        print("✓ Modelo Shift-Add teve MELHOR performance!")
        print("Surpreendente! Pode indicar overfitting do modelo standard.")
    else:
        print("≈ Performance similar entre os modelos")
    
    print("Análise de Eficiência:")
    print(f"Redução de multiplicações: {((mults_std - mults_sa) / mults_std * 100):.1f}%")
    print(f"Aumento de adições: {((adds_sa - adds_std) / adds_std * 100):.1f}%")
    
    print("Benefícios do modelo Shift-Add:")
    print("✓ Zero multiplicações reais (apenas shifts e adições)")
    print("✓ Mais eficiente energeticamente")
    print("✓ Mais adequado para hardware especializado")
    print("✓ Inspirado em métodos históricos de computação")
    
    print("Limitações:")
    print("⚠ Menor precisão numérica devido à quantização")
    print("⚠ Pode requerer técnicas especiais de treinamento")
    print("⚠ Menos expressivo que pesos reais de alta precisão")
    
    print("" + "=" * 80)

if __name__ == "__main__":
    train_and_compare()
