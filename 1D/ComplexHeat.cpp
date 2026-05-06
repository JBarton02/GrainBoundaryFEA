/*
 *
 */

#include "mfem.hpp"
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mfem/config/config.hpp>
#include <mfem/fem/bilinearform.hpp>
#include <mfem/fem/bilininteg.hpp>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/complex_fem.hpp>
#include <mfem/fem/fe_coll.hpp>
#include <mfem/fem/fespace.hpp>
#include <mfem/fem/gridfunc.hpp>
#include <mfem/fem/linearform.hpp>
#include <mfem/general/array.hpp>
#include <mfem/linalg/complex_operator.hpp>
#include <mfem/linalg/ode.hpp>
#include <mfem/linalg/operator.hpp>
#include <mfem/linalg/solvers.hpp>
#include <mfem/linalg/sparsemat.hpp>
#include <mfem/linalg/sparsesmoothers.hpp>
#include <mfem/linalg/vector.hpp>

using namespace std;
using namespace mfem;

bool check_for_inline_mesh(const char *mesh_file);

real_t InitialTemperature(const Vector &x);

// Material Properties
static real_t c_p = 1.6e6; // Heat Capacity [J/m^3-K]
static real_t k = 1.3;     // Thermal conductivity [W/m-K]
static real_t k_grain =
    1e-6; // Express a grain boundary with a very small conductivity [W/m-K]
static real_t rho = 2200; // Density of SiO_2 [Kg/m^3]
static real_t alpha =
    k / (c_p * rho); // Thermal diffusivity of material properties [m^2/s]
static real_t alpha_grain = k_grain / (c_p * rho);
static real_t omega_ = 1.0;

class ConductionOperator : public TimeDependentOperator {
protected:
  FiniteElementSpace &fespace;
  Array<int> ess_bdr;
  Array<int> ess_tdof_list; // should remain empty to simulate Neumann B.C.

  BilinearForm *M;
  BilinearForm *K;

  SparseMatrix Mmat, Kmat;
  SparseMatrix *T; // T = M + dt K
  real_t current_dt;

  CGSolver M_solver; // Krylov solver for inverting the mass matrix M
  DSmoother M_prec;  // Preconditioner for the mass matrix M

  CGSolver T_solver; // Implicit solver for T = M + dt K
  DSmoother T_prec;  // Preconditioner for the implicit solver

  real_t alpha; // Diffusive coefficient

  mutable Vector z; // Auxillary Vector

public:
  ConductionOperator(FiniteElementSpace &f, real_t alpha, const Vector &u);

  void Mult(const Vector &u, Vector &du_dt) const override;

  /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
  void setParameters(const Vector &u);

  ~ConductionOperator() override;
};

real_t InitialTemperature(const Vector &x);

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file = "2D_Grain.msh";
  int ref_levels = 1;
  int order = 2;

  real_t freq = -1.0;
  real_t heatFlux = 500;

  real_t t_final = 1.0;
  real_t dt = 1e-2;

  bool herm_conv = true;
  bool pa = false;

  string source_name = "Heat Flux";
  string ess_name = "Grain Boundary";
  const char *device_config = "cpu";

  bool visualization = 1;
  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&alpha, "-a", "--alpha", "Alpha coefficient.");
  args.AddOption(&freq, "-f", "--frequency", "Frequency (in Hz).");
  args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                 "--no-hermitian", "Use convention for Hermitian operators.");
  args.AddOption(&source_name, "-src", "--source-attr-name",
                 "Name of attribute set containing source.");
  args.AddOption(&ess_name, "-ess", "--ess-attr-name",
                 "Name of attribute set containing essential BC.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                 "--no-partial-assembly", "Enable Partial Assembly.");
  args.Parse();

  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  if (freq > 0.0) {
    omega_ = 2.0 * M_PI * freq;
  }

  ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes
  //    with the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  unique_ptr<ODESolver> ode_solver = ODESolver::Select(23);

  // 4. Refine the mesh to increase resolution. In this example we do
  //    'ref_levels' of uniform refinement where the user specifies
  //    the number of levels with the '-r' option.
  for (int l = 0; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }

  AttributeSets &attr_sets = mesh->attribute_sets;
  AttributeSets &bdr_attr_sets = mesh->bdr_attribute_sets;
  {
    std::set<string> names = attr_sets.GetAttributeSetNames();
    cout << "Element Attribute Set Names: ";
    for (auto const &set_name : names) {
      cout << " \"" << set_name << "\"";
    }
    cout << endl;

    std::set<string> bdr_names = bdr_attr_sets.GetAttributeSetNames();
    cout << "Boundary Attribute Set Names: ";
    for (auto const &bdr_set_name : bdr_names) {
      cout << " \"" << bdr_set_name << "\"";
    }
    cout << endl;
  }

  Array<int> &Grain = attr_sets.GetAttributeSet("Grain");
  Array<int> &Material = attr_sets.GetAttributeSet("Convection");
  Array<int> &HeatFlux = attr_sets.GetAttributeSet("HeatFlux");

  attr_sets.SetAttributeSet("Heat Flux", HeatFlux);
  attr_sets.SetAttributeSet("Grain", Grain);

  Array<int> &GrainBoundary = bdr_attr_sets.GetAttributeSet("ContactRes");

  bdr_attr_sets.SetAttributeSet("Grain Boundary", GrainBoundary);

  // Define a finite element space on the mesh using Lagrange finite
  // elements

  H1_FECollection fec(order, dim);
  FiniteElementSpace fespace(mesh, &fec);

  cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
       << endl;

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined based on the type
  //    of mesh and the problem type.
  Array<int> ess_tdof_list;
  Array<int> ess_bdr;

  if (bdr_attr_sets.AttributeSetExists(ess_name)) {

    ess_bdr.SetSize(mesh->bdr_attributes.Max());
    Array<int> ess_bdr_marker = bdr_attr_sets.GetAttributeSet(ess_name);
    fespace.GetEssentialTrueDofs(ess_bdr_marker, ess_tdof_list);

  } else if (mesh->bdr_attributes.Size()) {
    ess_bdr.SetSize(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  Array<int> source_marker = attr_sets.GetAttributeSetMarker(source_name);

  // 7. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system.
  ComplexLinearForm b(&fespace, conv);
  LinearForm b_real(&fespace, &b.real());
  LinearForm b_imag(&fespace, &b.imag());
  b = 0.0;
  b_real = 0.0;
  b_imag = 0.0;

  // 8. Define the solution vector u as a complex finite element grid function
  //    corresponding to fespace. Initialize u with initial guess of 1+0i or
  //    the exact solution if it is known.
  ComplexGridFunction u(&fespace);

  ConstantCoefficient thermalDiffusivity(alpha);
  ConstantCoefficient grainDiffusivity(alpha_grain);
  ConstantCoefficient thermalFlux(heatFlux);
  ProductCoefficient m_nbcCoef(alpha, thermalFlux);

  ConstantCoefficient zeroCoef(0.0);
  ConstantCoefficient oneCoef(1.0);

  Vector zeroVec(dim);
  zeroVec = 0.0;
  Vector oneVec(dim);
  oneVec = alpha;

  VectorConstantCoefficient zeroVecCoef(zeroVec);
  VectorConstantCoefficient oneVecCoef(oneVec);

  u.ProjectBdrCoefficient(oneCoef, zeroCoef, ess_bdr);

  Vector u_real = u.real();
  Vector u_complex = u.imag();
  ConductionOperator oper_real(fespace, alpha, u_real);
  ConductionOperator oper_imag(fespace, alpha, u_complex);

  SesquilinearForm *a = new SesquilinearForm(&fespace, conv);
  a->AddDomainIntegrator(new DiffusionIntegrator(thermalDiffusivity), NULL);
  a->AddDomainIntegrator(new DiffusionIntegrator(grainDiffusivity), NULL);

  BilinearForm *pcOp = new BilinearForm(&fespace);

  pcOp->AddDomainIntegrator(new DiffusionIntegrator(thermalDiffusivity));
  pcOp->AddDomainIntegrator(new DiffusionIntegrator(grainDiffusivity));

  // 10. Assemble the form and the corresponding linear system, applying any
  //     necessary transformations such as: assembly, eliminating boundary
  //     conditions, conforming constraints for non-conforming AMR, etc.
  a->Assemble();
  pcOp->Assemble();

  OperatorHandle A;
  Vector B, U;

  a->FormLinearSystem(ess_tdof_list, u, b, A, U, B);

  cout << "Size of linear system: " << A->Width() << endl << endl;

  // 11. Define and apply a GMRES solver for AU=B with a block diagonal
  //     preconditioner based on the appropriate sparse smoother.
  {
    Array<int> blockOffsets;
    blockOffsets.SetSize(3);
    blockOffsets[0] = 0;
    blockOffsets[1] = A->Height() / 2;
    blockOffsets[2] = A->Height() / 2;
    blockOffsets.PartialSum();

    BlockDiagonalPreconditioner BDP(blockOffsets);

    Operator *pc_r = NULL;
    Operator *pc_i = NULL;

    if (pa) {
      pc_r = new OperatorJacobiSmoother(*pcOp, ess_tdof_list);
    } else {
      OperatorHandle PCOp;
      pcOp->SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
      pcOp->FormSystemMatrix(ess_tdof_list, PCOp);
      pc_r = new DSmoother(*PCOp.As<SparseMatrix>());
    }
    real_t s = 1.0;
    pc_i =
        new ScaledOperator(pc_r, (conv == ComplexOperator::HERMITIAN) ? s : -s);

    BDP.SetDiagonalBlock(0, pc_r);
    BDP.SetDiagonalBlock(1, pc_i);
    BDP.owns_blocks = 1;

    GMRESSolver gmres;
    gmres.SetPreconditioner(BDP);
    gmres.SetOperator(*A.Ptr());
    gmres.SetRelTol(1e-12);
    gmres.SetMaxIter(1000);
    gmres.SetPrintLevel(1);
    gmres.Mult(B, U);
  }

  // 12. Recover the solution as a finite element grid function and compute the
  //     errors if the exact solution is known.
  a->RecoverFEMSolution(U, b, u);

  // 13. Save the refined mesh and the solution. This output can be viewed
  //     later using GLVis: "glvis -m mesh -g sol".
  {
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);

    ofstream sol_r_ofs("sol_r.gf");
    ofstream sol_i_ofs("sol_i.gf");
    ofstream sol_z_ofs("sol_z.gf");
    sol_r_ofs.precision(8);
    sol_i_ofs.precision(8);
    sol_z_ofs.precision(8);
    u.real().Save(sol_r_ofs);
    u.imag().Save(sol_i_ofs);
    u.Save(sol_z_ofs);
  }

  // 14. Send the solution by socket to a GLVis server.
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock_r(vishost, visport);
    socketstream sol_sock_i(vishost, visport);
    sol_sock_r.precision(8);
    sol_sock_i.precision(8);
    sol_sock_r << "solution\n"
               << *mesh << u.real() << "window_title 'Solution: Real Part'"
               << flush;
    sol_sock_i << "solution\n"
               << *mesh << u.imag() << "window_title 'Solution: Imaginary Part'"
               << flush;
  }
  if (visualization) {
    GridFunction u_t(&fespace);
    u_t = u.real();
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n"
             << *mesh << u_t << "window_title 'Harmonic Solution (t = 0.0 T)'"
             << "pause\n"
             << flush;

    cout << "GLVis visualization paused."
         << " Press space (in the GLVis window) to resume it.\n";
    int num_frames = 32;
    int i = 0;
    while (sol_sock) {
      real_t t = (real_t)(i % num_frames) / num_frames;
      ostringstream oss;
      oss << "Harmonic Solution (t = " << t << " T)";

      add(cos(2.0 * M_PI * t), u.real(), sin(-2.0 * M_PI * t), u.imag(), u_t);
      sol_sock << "solution\n"
               << *mesh << u_t << "window_title '" << oss.str() << "'" << flush;
      i++;
    }
  }

  // 15. Free the used memory.
  delete a;
  delete pcOp;
  delete &fespace;
  delete &fec;
  delete mesh;

  return 0;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, real_t al,
                                       const Vector &u)
    : TimeDependentOperator(f.GetTrueVSize(), (real_t)0.0, EXPLICIT),
      fespace(f), M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height) {
  const real_t rel_tol = 1e-8;

  M = new BilinearForm(&fespace);
  M->AddDomainIntegrator(new MassIntegrator());
  M->Assemble();
  M->FormSystemMatrix(ess_tdof_list, Mmat);

  M_solver.iterative_mode = false;
  M_solver.SetRelTol(rel_tol);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(30);
  M_solver.SetPrintLevel(0);
  M_solver.SetPreconditioner(M_prec);
  M_solver.SetOperator(Mmat);

  alpha = al;

  T_solver.iterative_mode = false;
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(100);
  T_solver.SetPrintLevel(0);
  T_solver.SetPreconditioner(T_prec);

  setParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const {
  // Compute:
  //    du_dt = M^{-1}*-Ku
  // for du_dt, where K is linearized by using u from the previous timestep
  Kmat.Mult(u, z);
  z.Neg(); // z = -z
  M_solver.Mult(z, du_dt);
}

void ConductionOperator::setParameters(const Vector &u) {
  GridFunction u_alpha_gf(&fespace);
  u_alpha_gf.SetFromTrueDofs(u);
  for (int i = 0; i < u_alpha_gf.Size(); i++) {
    u_alpha_gf(i) = alpha * u_alpha_gf(i);
  }

  delete K;
  K = new BilinearForm(&fespace);

  GridFunctionCoefficient u_coeff(&u_alpha_gf);

  K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
  K->Assemble();
  K->FormSystemMatrix(ess_tdof_list, Kmat);
  delete T;
  T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator() {
  delete T;
  delete M;
  delete K;
}

bool check_for_inline_mesh(const char *mesh_file) {
  string file(mesh_file);
  size_t p0 = file.find_last_of("/");
  string s0 = file.substr((p0 == string::npos) ? 0 : (p0 + 1), 7);
  return s0 == "inline-";
}
real_t InitialTemperature(const Vector &x) {
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}
