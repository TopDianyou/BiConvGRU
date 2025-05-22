import torch
from torch import nn
from typing import List, Optional, Tuple, Union


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
    ):
        super().__init__()
        self.height, self.width = input_size
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.sigmoid(gates).chunk(2, dim=1)

        combined_candidate = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_candidate))

        h_next = (1 - update_gate) * h_prev + update_gate * candidate
        return h_next


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        input_dim: Union[int, List[int]],
        hidden_dim: Union[int, List[int]],
        kernel_size: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]],
        num_layers: int,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
        dropout: float = 0.0,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.num_layers = num_layers

        self.input_dim = self._expand_param(input_dim, num_layers)
        self.hidden_dim = self._expand_param(hidden_dim, num_layers)
        self.kernel_size = self._expand_param(kernel_size, num_layers)

        self.downsample = nn.Identity()
        if output_size is not None and output_size != input_size:
            self.downsample = nn.AdaptiveAvgPool2d(output_size)
            self.input_size = output_size

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = self.input_dim[i] if i == 0 else self.hidden_dim[i - 1]
            self.cells.append(
                ConvGRUCell(
                    input_size=self.input_size,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=bias,
                )
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, _, _ = x.size()
        device = x.device

        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        x = self.downsample(x)
        _, C, H, W = x.size()
        x = x.view(batch_size, seq_len, C, H, W)

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, device)

        layer_outputs = []
        last_states = []
        cur_input = x

        for layer_idx, cell in enumerate(self.cells):
            h = hidden_state[layer_idx]
            outputs = []
            for t in range(seq_len):
                h = cell(cur_input[:, t], h)
                outputs.append(h)
            outputs = torch.stack(outputs, dim=1)
            outputs = self.dropout(outputs)
            layer_outputs.append(outputs)
            last_states.append(h)
            cur_input = outputs

        if not self.return_all_layers:
            layer_outputs = layer_outputs[-1:]
            last_states = last_states[-1:]

        return layer_outputs, last_states

    def _init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [cell.init_hidden(batch_size, device) for cell in self.cells]

    @staticmethod
    def _expand_param(param, num_layers):
        if isinstance(param, list):
            assert len(param) == num_layers, "list param length must match num_layers"
            return param
        else:
            return [param] * num_layers


class BiConvGRU(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        input_dim: Union[int, List[int]],
        hidden_dim: Union[int, List[int]],
        kernel_size: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]],
        num_layers: int,
        bidirectional: bool = True,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
        dropout: float = 0.0,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.forward_gru = ConvGRU(
            input_size=input_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bias=bias,
            return_all_layers=return_all_layers,
            dropout=dropout,
            output_size=output_size,
        )
        if self.bidirectional:
            self.reverse_gru = ConvGRU(
                input_size=input_size,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bias=bias,
                return_all_layers=return_all_layers,
                dropout=dropout,
                output_size=output_size,
            )

    def forward(
        self, x: torch.Tensor, return_sequence: bool = True
    ) -> Tuple[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if not self.forward_gru.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        fwd_out, fwd_state = self.forward_gru(x)

        if self.bidirectional:
            rev_idx = torch.arange(x.size(1) - 1, -1, -1, device=x.device)
            x_rev = x.index_select(1, rev_idx)
            rev_out, rev_state = self.reverse_gru(x_rev)
            rev_out = [o.index_select(1, rev_idx) for o in rev_out]

            outputs = [torch.cat([f, r], dim=2) for f, r in zip(fwd_out, rev_out)]
            states = (fwd_state, rev_state)
        else:
            outputs = fwd_out
            states = (fwd_state,)

        if not return_sequence:
            outputs = [o[:, -1] for o in outputs]

        if not self.forward_gru.return_all_layers:
            outputs = outputs[-1:]
            states = tuple(s[-1:] for s in states)

        return outputs, states


if __name__ == "__main__":
    device = torch.device("cpu")

    model = BiConvGRU(
        input_size=(64, 64),
        input_dim=3,
        hidden_dim=[32, 32],
        kernel_size=[3, 3],
        num_layers=2,
        bidirectional=True,
        batch_first=True,
        dropout=0.1,
        output_size=(64, 64),
    ).to(device)

    inputs = torch.randn(4, 8, 3, 64, 64).to(device)

    outputs, states = model(inputs)

    print("shape:", outputs[0].shape)  # (4, 8, 64, 32, 32)
    print("shape:", states[0][-1].shape)  # (4, 32, 32, 32)