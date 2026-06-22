import Nav from './components/Nav'
import Hero from './components/Hero'
import Datasets from './components/Datasets'
import Preprocessing from './components/Preprocessing'
import GeneticFeatures from './components/GeneticFeatures'
import Ctgan from './components/Ctgan'
import Training from './components/Training'
import Fusion from './components/Fusion'
import LiveDemo from './components/LiveDemo'
import Results from './components/Results'
import Footer from './components/Footer'

function Divider() {
  return (
    <div className="max-w-7xl mx-auto px-8">
      <div className="h-px bg-gradient-to-r from-transparent via-white/[0.08] to-transparent" />
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-screen">
      <Nav />
      <main>
        <Hero />
        <Divider />
        <Datasets />
        <Divider />
        <Preprocessing />
        <Divider />
        <GeneticFeatures />
        <Divider />
        <Ctgan />
        <Divider />
        <Training />
        <Divider />
        <Fusion />
        <Divider />
        <LiveDemo />
        <Divider />
        <Results />
      </main>
      <Footer />
    </div>
  )
}
