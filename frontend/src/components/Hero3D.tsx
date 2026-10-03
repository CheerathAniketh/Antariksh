import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Stars } from '@react-three/drei'
import * as THREE from 'three'
import { animTime, planetAngle } from '../lib/orbit'
import type { OrbitParams } from '../lib/orbit'

function Scene({ orbit }: { orbit: OrbitParams }) {
  const planet = useRef<THREE.Mesh>(null)
  const camZ = Math.max(16, orbit.a * 2.3)
  const planetR = Math.max(orbit.rp, 0.1)

  useFrame(({ camera }) => {
    const th = planetAngle(animTime(), orbit)
    if (planet.current) {
      planet.current.position.set(orbit.a * Math.sin(th), 0, orbit.a * Math.cos(th))
    }
    camera.position.set(0, 0.5, camera.position.z + (camZ - camera.position.z) * 0.05)
    camera.lookAt(0, 0, 0)
  })

  return (
    <>
      <ambientLight intensity={0.35} />
      <pointLight position={[0, 0, 0]} intensity={300} decay={2} color="#ffd9a8" />
      <Stars radius={80} depth={40} count={5000} factor={3.5} fade speed={0.6} />

      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshBasicMaterial color="#ffe2b0" toneMapped={false} />
      </mesh>
      {[1.25, 1.6, 2.2].map((r, i) => (
        <mesh key={r} scale={r}>
          <sphereGeometry args={[1, 32, 32]} />
          <meshBasicMaterial
            color="#ffb86b"
            transparent
            opacity={0.16 / (i + 1)}
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

      <mesh ref={planet}>
        <sphereGeometry args={[planetR, 32, 32]} />
        <meshStandardMaterial color="#6fa8ff" roughness={0.8} />
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
