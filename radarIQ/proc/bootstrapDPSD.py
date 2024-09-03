import numpy as np

def bootstrapDPSD(V, w, N0, NFT, B, K, N):
    r = 0.5 - np.sqrt(np.mean(np.power(w, 2)))*0.5

    NK = V['H'].shape[0]
    M = V['H'].shape[1]

    if NFT is None:
        NFT = M

    S = {
        'H': np.full((NK, NFT), np.nan),
        'V': np.full((NK, NFT), np.nan),
        'X': np.full((NK, NFT), np.nan + np.nan * 1j)
    }

    for i in range(NK):
        tV = {
            'H': V['H'][i,:],
            'V': V['V'][i,:]
        }
        
        CX_left = {
            'H': 0.5*(tV['H'][0]/tV['H'][-1] + tV['V'][0]/tV['V'][-1]),
            'V': 0.5*(tV['H'][0]/tV['H'][-1] + tV['V'][0]/tV['V'][-1])
        }
        CX_right = {
            'H': 0.5*(tV['H'][-1]/tV['H'][0] + tV['V'][-1]/tV['V'][0]),
            'V': 0.5*(tV['H'][-1]/tV['H'][0] + tV['V'][-1]/tV['V'][0])
        }

        VL = {
            'H': tV['H'][-round(M * r):-1] * CX_left['H'],
            'V': tV['V'][-round(M * r):-1] * CX_left['V']
        }
        VR = {
            'H': tV['H'][1:round(M * r)] * CX_right['H'],
            'V': tV['V'][1:round(M * r)] * CX_right['V'],
        }

        X = {
            'H': np.concatenate((VL['H'], tV['H'], VR['H'])),
            'V': np.concatenate((VL['V'], tV['V'], VR['V']))
        }

        Mx = len(X['H'])
        boot_indexes = np.random.randint(0, Mx - M + 1, size=(B, 1))
        boot_indexes = boot_indexes + np.tile(np.arange(0, M, 1), (B, 1))

        tV = {
            'H': X['H'][boot_indexes],
            'V': X['V'][boot_indexes]
        }

        R0 = {
            'H': np.mean(V['H'][i,:] * np.conjugate(V['H'][i,:])),
            'V': np.mean(V['V'][i,:] * np.conjugate(V['V'][i,:]))
        }

        tR0 = {
            'H': np.mean(tV['H'] * np.conjugate(tV['H']), axis=1),
            'V': np.mean(tV['V'] * np.conjugate(tV['V']), axis=1)
        }

        tV['H'] = np.array([np.sqrt(R0['H'] / tR0['H'])]).T * tV['H']
        tV['V'] = np.array([np.sqrt(R0['V'] / tR0['V'])]).T * tV['V']

        z = {
            'H': np.fft.fftshift(np.fft.fft(tV['H']*w, n=NFT, axis=1), axes=1),
            'V': np.fft.fftshift(np.fft.fft(tV['V']*w, n=NFT, axis=1), axes=1)
        }

        alpha = np.mean(np.power(np.abs(w), 2))
        S['H'][i,:] = np.mean((np.power(np.abs(z['H']), 2)) / (M * alpha), axis=0)
        S['V'][i,:] = np.mean((np.power(np.abs(z['V']), 2)) / (M * alpha), axis=0)
        S['X'][i,:] = np.mean((z['H'] * np.conjugate(z['V'])) / (M * alpha), axis=0)

    tsh = np.full((N, NFT), np.nan)
    tsv = np.full((N, NFT), np.nan)
    tsx = np.full((N, NFT), np.nan + np.nan * 1j)
    td = np.full((N, NFT), np.nan)
    tr = np.full((N, NFT), np.nan)

    for i in range(N):
        iK = np.arange(0, K+1, 1) + (i-1)*K
        
        tsh[i,:] = np.mean(S['H'][iK,:], axis=0)
        tsv[i,:] = np.mean(S['V'][iK,:], axis=0)
        tsx[i,:] = np.mean(S['X'][iK,:], axis=0)

        td[i,:] = tsh[i,:] / tsv[i,:]
        tr[i,:] = np.abs(tsx[i,:]) / np.sqrt(tsh[i,:] * tsv[i,:])

    if K == 1:
        beta = (1-r)**(-3.3) - 2*((1-r)**1.1)
    else:
        beta = (1-r)**(-4.5) - (1-r)**(-2.1)

    E = {}
    E['sS'] = {
        'H': tsh,
        'V': tsv,
        'X': tsx
    }
    E['sSNR'] = {
        'H': tsh / N0['H'],
        'V': tsv / N0['V']
    }
    E['sD'] = td * (1 - (1 / (beta * K) * (1 - np.power(tr, 2))))
    E['sR'] = tr * (1 - (1 / (beta * K) * ((np.power(1 - np.power(tr, 2), 2)) / (4 * np.power(tr, 2)))))

    E['sD'][E['sD'] < 0] = np.nan
    E['sR'][E['sR'] < 0] = 0

    return E