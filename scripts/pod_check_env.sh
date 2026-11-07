#!/bin/bash
echo "--- setup process ---"
ps -o pid,etime,cmd -C bash 2>/dev/null | grep -i setup || echo "no setup bash running"
echo "--- log tail ---"
tail -n 12 /root/setup.log
echo "--- torch ---"
/local/ovi-env/bin/python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -5
echo "--- torchaudio ---"
/local/ovi-env/bin/python3 -c "import torchaudio; print('torchaudio', torchaudio.__version__)" 2>&1 | tail -5
echo "--- flash_attn ---"
/local/ovi-env/bin/python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1 | tail -5
