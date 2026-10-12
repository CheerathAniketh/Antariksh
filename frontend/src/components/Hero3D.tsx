import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Stars } from '@react-three/drei'
import * as THREE from 'three'
import { animTime, fluxAt, planetAngle } from '../lib/orbit'
import type { OrbitParams } from '../lib/orbit'

const TRAIL_LEN = 90
const GLOW_BASE = [0.16, 0.08, 0.053]

function Scene({ orbit }: { orbit: OrbitParams }) {
  const planet = useRef<THREE.Mesh>(null)
  const starMat = useRef<THREE.MeshBasicMaterial>(null)
  const glowMats = useRef<(THREE.MeshBasicMaterial | null)[]>([])
  const starfield = useRef<THREE.Group>(null)
  const camZ = Math.max(16, orbit.a * 2.3)
  const planetR = Math.max(orbit.rp, 0.1)

  const trail = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3))
    const col = new Float32Array(TRAIL_LEN * 3)
    for (let i = 0; i < TRAIL_LEN; i++) {
      const f = 1 - i / (TRAIL_LEN - 1)
      col[i * 3] = 0.37 * f
      col[i * 3 + 1] = 0.9 * f
      col[i * 3 + 2] = f
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3))
    const m = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
    return new THREE.Line(g, m)
  }, [])

  useFrame(({ camera, pointer }) => {
    const t = animTime()
    const th = planetAngle(t, orbit)
    planet.current?.position.set(orbit.a * Math.sin(th), 0, orbit.a * Math.cos(th))

    const pos = trail.geometry.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < TRAIL_LEN; i++) {
      const a = th - i * 0.03
      pos.setXYZ(i, orbit.a * Math.sin(a), 0, orbit.a * Math.cos(a))
    }
    pos.needsUpdate = true

    // star dims during transit; exaggerated (35%) so it is visible
    const depth = orbit.rp * orbit.rp
    const progress = depth > 0 ? Math.min(1, Math.max(0, (1 - fluxAt(t, orbit)) / depth)) : 0
    const dim = 1 - 0.35 * progress
    starMat.current?.color.setRGB(dim, 0.886 * dim, 0.69 * dim)
    glowMats.current.forEach((m, i) => {
      if (m) m.opacity = GLOW_BASE[i] * dim
    })

    if (starfield.current) starfield.current.rotation.y = t * 0.01

    camera.position.x += (pointer.x * 1.2 - camera.position.x) * 0.04
    camera.position.y += (0.5 + pointer.y * 0.6 + Math.sin(t * 0.15) * 0.15 - camera.position.y) * 0.04
    camera.position.z += (camZ - camera.position.z) * 0.05
    camera.lookAt(0, 0, 0)
  })

  return (
    <>
      <ambientLight intensity={0.35} />
      <pointLight position={[0, 0, 0]} intensity={300} decay={2} color="#ffd9a8" />
      <group ref={starfield}>
        <Stars radius={80} depth={40} count={5000} factor={3.5} fade speed={0.6} />
      </group>

      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshBasicMaterial ref={starMat} color="#ffe2b0" toneMapped={false} />
      </mesh>
      {[1.25, 1.6, 2.2].map((r, i) => (
        <mesh key={r} scale={r}>
          <sphereGeometry args={[1, 32, 32]} />
          <meshBasicMaterial
            ref={(m) => {
              glowMats.current[i] = m
            }}
            color="#ffb86b"
            transparent
            opacity={GLOW_BASE[i]}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.BackSide}
          />
        </mesh>
      ))}

      <mesh rotation-x={Math.PI / 2}>
        <ringGeometry args={[orbit.a - 0.015, orbit.a + 0.015, 160]} />
        <meshBasicMaterial color="#3a5a7a" transparent opacity={0.55} side={THREE.DoubleSide} />
      </mesh>

      <primitive object={trail} />

      <mesh ref={planet}>
        <sphereGeometry args={[planetR, 32, 32]} />
        <meshStandardMaterial color="#6fa8ff" roughness={0.8} />
        <mesh scale={1.35}>
          <sphereGeometry args={[1, 24, 24]} />
          <meshBasicMaterial
            color="#5ee7ff"
            transparent
            opacity={0.18}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.BackSide}
          />
        </mesh>
      </mesh>
    </>
  )
}

export default function Hero3D({ orbit }: { orbit: OrbitParams }) {
  return (
    <Canvas camera={{ position: [0, 0.5, 16], fov: 35 }} dpr={[1, 2]}>
      <Scene orbit={orbit} />
    </Canvas>
  )
}
